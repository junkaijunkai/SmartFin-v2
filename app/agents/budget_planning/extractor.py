from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from app.config import resolve_model_name, get_prompt
from app.state import TransactionCategory
from app.tools.cache import get_cached_llm_response, cache_llm_response


logger = logging.getLogger(__name__)

MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds

# All valid spending categories
SUPPORTED_CATEGORIES = [cat.value for cat in TransactionCategory if cat.value != "income"]


class BudgetRequest(BaseModel):
    intent: str = Field(default="budget_planning")
    user_message: str
    monthly_income: Optional[float] = None
    categories_requested: List[str] = Field(default_factory=list)
    needs_clarification: bool = False


def extract_budget_request(
    state: Dict[str, Any],
    context: str | None = None,
) -> Dict[str, Any]:
    """
    Extract a structured budget-planning request from shared state via LLM.

    context: 可选，对话上下文。仅注入 LLM system prompt，不参与 fallback。

    Responsibilities:
    - read latest user message
    - call LLM to normalize the request into structured fields
    - fall back to state["monthly_income"] if the user did not mention income
    """

    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""

    # fallback income from state
    state_income = state.get("monthly_income")

    model_name = resolve_model_name("default")

    try:
        llm = ChatAnthropic(model=model_name, temperature=0, timeout=_LLM_TIMEOUT)
        structured_llm = llm.with_structured_output(BudgetRequest)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for budget extraction: %s", exc)
        return {
            "intent": "budget_planning",
            "user_message": last_message,
            "monthly_income": state_income,
            "categories_requested": [],
            "needs_clarification": state_income is None,
        }

    prompt_context = (
        f"Continuing conversation — context from previous turn: {context}\n\n"
        if context else ""
    )
    messages = get_prompt("budget_request_extractor").format_messages(
        context=prompt_context,
        supported_categories=", ".join(SUPPORTED_CATEGORIES),
        last_message=last_message,
        state_income=str(state_income) if state_income is not None else "unknown",
    )

    cache_key = f"{last_message}|{str(state_income)}|{context or ''}"
    cached = get_cached_llm_response("budget_request_extractor", cache_key)
    if cached is not None:
        logger.debug("[budget_extractor] cache hit")
        return cached

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = structured_llm.invoke(messages)
            monthly_income = result.monthly_income if result.monthly_income is not None else state_income
            output = {
                "intent": "budget_planning",
                "user_message": result.user_message,
                "monthly_income": monthly_income,
                "categories_requested": result.categories_requested,
                "needs_clarification": monthly_income is None,
            }
            cache_llm_response("budget_request_extractor", cache_key, output)
            return output
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "Budget extraction failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Budget extraction failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    return {
        "intent": "budget_planning",
        "user_message": last_message,
        "monthly_income": state_income,
        "categories_requested": [],
        "needs_clarification": state_income is None,
    }