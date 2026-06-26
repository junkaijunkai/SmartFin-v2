"""
Goal Planning extractor — uses Claude to understand the user's latest message
and extract structured financial goal information.

Design principle:
- LLM is used only for language understanding / field extraction
- deterministic financial calculations remain in tracker.py
- testing concerns should stay in the test layer, not in production code
"""

from __future__ import annotations

import calendar
import logging
import os
import re
import time
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.config import get_llm, get_prompt
from app.tools.cache import get_cached_llm_response, cache_llm_response

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

class GoalExtractionResult(BaseModel):
    """
    结构化提取结果。

    字段说明：
    - is_goal_intent: 用户是否表达了一个“财务目标 / 储蓄目标”
    - name: 目标名称，例如 "Laptop Fund"
    - target_amount: 目标金额
    - target_date: 目标截止日期
    - current_amount: 当前已存金额（如果用户提到了）
    - missing_fields: 创建目标仍缺失的必要字段
    """
    is_goal_intent: bool = Field(
        description="True if the user is expressing or discussing a financial savings goal."
    )
    name: Optional[str] = Field(
        default=None,
        description="A concise goal name such as 'Laptop Fund' or 'Emergency Fund'."
    )
    target_amount: Optional[float] = Field(
        default=None,
        description="The target amount the user wants to accumulate."
    )
    target_date: Optional[date] = Field(
        default=None,
        description=(
            "The target date by which the user wants to reach the goal. "
            "MUST be an exact YYYY-MM-DD date. Resolve all relative expressions "
            "(e.g. 'end of this month', 'by June', 'in 3 months', 'next year') "
            "based on today's date provided in the system prompt."
        ),
    )
    current_amount: Optional[float] = Field(
        default=None,
        description="The amount already saved toward this goal, if mentioned."
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Missing required fields, typically target_amount and/or target_date."
    )
    is_update_intent: bool = Field(
        default=False,
        description=(
            "True if the user wants to adjust, update, modify, or change an existing goal "
            "rather than create a new one. The 'name' field should contain the name of the "
            "goal they want to modify."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Normalization — ensures missing_fields is always correct regardless of LLM
# ---------------------------------------------------------------------------


def _normalize_missing_fields(result: GoalExtractionResult) -> None:
    """
    Deterministically recompute missing_fields from the extracted values.

    The LLM may not reliably populate missing_fields (especially smaller
    models).  This ensures the field is always correct based on what was
    actually extracted.  Only applied to create intent — update intent
    does not require all fields.
    """
    if not result.is_goal_intent or result.is_update_intent:
        return
    computed: list[str] = []
    if result.target_amount is None:
        computed.append("target_amount")
    if result.target_date is None:
        computed.append("target_date")
    result.missing_fields = computed


# ---------------------------------------------------------------------------
# Fallback extraction
# ---------------------------------------------------------------------------

# 提取数字金额，支持货币符号前缀和千位逗号，例如 $8,000 / £1,200.50 / 640000
_CURRENCY_PREFIX = re.compile(r"[$£€¥₹]")
_AMOUNT_PATTERN = re.compile(r"(\d[\d,]*(?:\.\d+)?)")

# 仅支持简单的 YYYY-MM-DD 格式日期
_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _extract_relative_date(msg_lower: str, today: date) -> Optional[date]:
    """Resolve common relative date expressions to a concrete date."""
    if re.search(r"end of next year|by next year|by the end of next year", msg_lower):
        return date(today.year + 1, 12, 31)
    if re.search(r"end of (this )?year|end of year|by year.?s? end|by end of year", msg_lower):
        return date(today.year, 12, 31)
    if re.search(r"end of (this )?month|end of month|by month.?s? end|by end of month|by the end of (this )?month", msg_lower):
        return date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
    m = re.search(r"in (\d+) months?", msg_lower)
    if m:
        target = today + relativedelta(months=int(m.group(1)))
        return date(target.year, target.month, calendar.monthrange(target.year, target.month)[1])
    if re.search(r"next month", msg_lower):
        target = today + relativedelta(months=1)
        return date(target.year, target.month, calendar.monthrange(target.year, target.month)[1])
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    month_alternation = "|".join(month_names)
    for pattern in [
        rf"\bby\s+the\s+end\s+of\s+({month_alternation})\b",
        rf"\bby\s+({month_alternation})\b",
    ]:
        m = re.search(pattern, msg_lower)
        if m:
            month_num = month_names[m.group(1)]
            year = today.year if month_num >= today.month else today.year + 1
            return date(year, month_num, calendar.monthrange(year, month_num)[1])
    return None


def _fallback_extract(
    user_message: str,
    today: Optional[date] = None,
    context: str | None = None,
) -> GoalExtractionResult:
    """
    当 LLM 调用失败时使用的轻量级兜底逻辑。

    这个 fallback 的目标不是“非常聪明”，而是：
    1. 能识别大致 goal intent
    2. 尽量提取金额和日期
    3. 给 agent 提供一个可继续处理的结构化结果
    """
    msg = user_message.lower()
    today = today or date.today()

    # Check for update intent before create intent
    update_keywords = [
        "adjust", "update", "modify", "change", "revise",
        "increase", "decrease", "edit",
    ]
    is_update_intent = any(keyword in msg for keyword in update_keywords)

    # 一组非常简单的关键词，用于粗粒度判断是否像是"财务目标"
    goal_keywords = [
        "save", "saving", "savings", "goal", "fund", "deposit",
        "emergency", "laptop", "travel", "holiday", "house",
        "accumulate", "set aside", "put aside",
    ]
    is_goal_intent = is_update_intent or any(keyword in msg for keyword in goal_keywords)

    # When continuing a goal conversation, treat the message as goal intent
    # even if it contains no goal keywords (e.g. a bare date fragment)
    if context and not is_goal_intent:
        is_goal_intent = True

    # ------------------------------------------------------------------
    # 先提取日期（ISO 格式优先，再尝试相对日期表达）
    # 这样后面提取金额时，可以先把日期字符串从文本里去掉，
    # 避免把 2027-06-01 中的年份 2027 误识别成 target_amount
    # ------------------------------------------------------------------
    extracted_date: Optional[date] = None
    message_without_date = user_message

    date_match = _DATE_PATTERN.search(user_message)
    if date_match:
        try:
            date_str = date_match.group(1)
            extracted_date = date.fromisoformat(date_str)
            message_without_date = user_message.replace(date_str, "")  # 去掉日期部分，避免干扰金额提取
        except ValueError:
            extracted_date = None

    if extracted_date is None:
        extracted_date = _extract_relative_date(msg, today)

    # ------------------------------------------------------------------
    # 再提取金额
    # 先去掉货币符号，再去掉千位逗号，避免 $64,0000 被截断为 64
    # 注意：使用”去掉日期后的文本”，避免年份被误识别为金额
    # ------------------------------------------------------------------
    extracted_amount: Optional[float] = None
    normalized = _CURRENCY_PREFIX.sub("", message_without_date)
    amount_match = _AMOUNT_PATTERN.search(normalized)
    if amount_match:
        try:
            extracted_amount = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            extracted_amount = None

    # 根据关键词给一个比较自然的目标名称
    goal_name: Optional[str] = None
    if "laptop" in msg:
        goal_name = "Laptop Fund"
    elif "emergency" in msg:
        goal_name = "Emergency Fund"
    elif "travel" in msg or "holiday" in msg:
        goal_name = "Travel Fund"
    elif "house" in msg or "deposit" in msg:
        goal_name = "House Deposit Fund"
    elif is_goal_intent:
        # 如果看起来像 goal intent，但识别不出具体类别
        goal_name = "Financial Goal"

    # 如果用户表达了 goal intent，但缺少必要字段，就记录缺失项
    missing_fields: list[str] = []
    if is_goal_intent and not is_update_intent:
        if extracted_amount is None:
            missing_fields.append("target_amount")
        if extracted_date is None:
            missing_fields.append("target_date")

    return GoalExtractionResult(
        is_goal_intent=is_goal_intent,
        name=goal_name,
        target_amount=extracted_amount,
        target_date=extracted_date,
        current_amount=None,
        missing_fields=missing_fields,
        is_update_intent=is_update_intent,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_goal_from_message(
    user_message: str,
    today: date | None = None,
    context: str | None = None,
) -> tuple[GoalExtractionResult, bool]:
    """
    从用户消息中提取结构化的 goal 信息。

    context: 可选，对话上下文。仅注入 LLM 的 system prompt，不影响
    fallback regex（避免上下文文本污染正则匹配）。

    返回：
        (result, llm_succeeded)
    """

    # 空输入时，直接返回一个”没有 goal intent”的结果
    if not user_message.strip():
        return GoalExtractionResult(
            is_goal_intent=False,
            missing_fields=[],
        ), True

    today = today or date.today()

    # 允许通过环境变量覆盖模型名，但不再在生产代码里放 mock mode
    try:
        llm = get_llm("planner", timeout=_LLM_TIMEOUT)
        structured_llm = llm.with_structured_output(GoalExtractionResult)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for goal extraction: %s", exc)
        return _fallback_extract(user_message, today=today, context=context), False

    # build prompt — context goes into the system prompt (LLM-only),
    # NOT into user_message (keeps regex fallback clean)
    prompt_context = (
        f"Continuing conversation — context from previous turn: {context}\n\n"
        if context else ""
    )
    messages = get_prompt("goal_extractor").format_messages(
        today=today.isoformat(),
        context=prompt_context,
        user_message=user_message,
    )

    cache_key = f"{user_message}|{today.isoformat()}|{context or ''}"
    cached = get_cached_llm_response("goal_extractor", cache_key)
    if cached is not None:
        logger.debug("[goal_extractor] cache hit")
        result = GoalExtractionResult(**cached)
        _normalize_missing_fields(result)
        return result, True

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result: GoalExtractionResult = structured_llm.invoke(messages)
            _normalize_missing_fields(result)
            cache_llm_response("goal_extractor", cache_key, result.model_dump(mode="json"))
            return result, True
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Goal extraction failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Goal extraction failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    return _fallback_extract(user_message, today=today, context=context), False