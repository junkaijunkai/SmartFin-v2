from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5"
_MODEL_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "model_registry.json"
_EXTRA_LOG_FIELDS = ("agent", "model", "thread_id", "guardrail", "event",
                     "trace_id", "span_id", "parent_span_id", "node")


# ---------------------------------------------------------------------------
# ReAct system prompt defaults (inline fallbacks when LangSmith unavailable)
# ---------------------------------------------------------------------------

_REACT_PROMPT_DEFAULTS: dict[str, str] = {
    "react_expense_analysis": (
        "You are the Expense Analysis Agent for the SmartFin financial assistant.\n\n"
        "Your job is to categorise raw transactions and compute spending trends.\n\n"
        "Available tools:\n"
        "  - categorise_transactions_tool: Takes a list of raw transaction dicts and "
        "returns them with spending categories assigned.\n"
        "  - compute_spending_trends_tool: Takes categorised transactions and computes "
        "30-day spending trends per category.\n\n"
        "Workflow:\n"
        "  1. You will receive transaction data in the user message.\n"
        "  2. First, call categorise_transactions_tool to categorise any raw transactions.\n"
        "  3. Then call compute_spending_trends_tool to analyse trends.\n"
        "  4. When both tools have returned results, write a concise summary of your "
        "findings as your final response.\n\n"
        "Guidelines:\n"
        "  - Your summary should highlight key findings, top spending categories, and "
        "notable changes versus the previous period.\n"
        "  - Scope boundary: You are ONLY the expense analysis agent. The user message "
        "may mention goals, budgets, or other financial topics — ignore them. Those "
        "will be handled by other agents. Only discuss expense categorisation and "
        "spending trends in your summary."
    ),
    "react_budget_planning": (
        "You are the Budget Planning Agent for the SmartFin financial assistant.\n\n"
        "Your job is to help users create and review budget plans based on their "
        "spending data and income.\n\n"
        "Available tools:\n"
        "  - extract_budget_request_tool: Record the monthly income and category "
        "preferences you have identified. Extract these values yourself from the user "
        "message and the context below before calling — do not pass null if the income "
        "is visible in the context.\n"
        "  - generate_allocations_tool: Create budget allocations for each spending "
        "category based on historical averages and trends.\n"
        "  - calculate_spending_tool: Compute actual monthly spending from transactions.\n"
        "  - evaluate_progress_tool: Compare actual spending against budget targets.\n"
        "  - generate_warnings_tool: Generate warnings for categories that may exceed budget.\n"
        "  - validate_budget_tool: Validate the complete budget output structure.\n\n"
        "Workflow:\n"
        "  1. Read the user's message and the context below. Extract monthly income "
        "(use context value if not stated in the message) and any category preferences.\n"
        "  2. Call extract_budget_request_tool with the extracted values.\n"
        "  3. If the tool reports income is missing, respond directly asking the user "
        "for their monthly income before proceeding.\n"
        "  4. Generate allocations using generate_allocations_tool with the spending data "
        "provided in context.\n"
        "  5. Calculate actual spending with calculate_spending_tool.\n"
        "  6. Evaluate progress with evaluate_progress_tool.\n"
        "  7. Generate warnings with generate_warnings_tool.\n"
        "  8. Validate with validate_budget_tool. If invalid, fix and re-validate.\n"
        "  9. When validation passes, respond with a clear summary of the budget plan.\n\n"
        "Context data (provided in the user message):\n"
        "  - Monthly income: {monthly_income}\n"
        "  - Categories and their spending data:\n{categories}\n"
        "  - Current period: {current_month}, day {current_day} of {days_in_month}\n\n"
        "Scope boundary: You are ONLY the budget planning agent. The user message may "
        "mention goals, anomalies, or other financial topics — ignore them. Only "
        "discuss budget allocations, spending progress, and warnings."
    ),
    "react_goal_planning": (
        "You are the Goal Planning Agent for the SmartFin financial assistant.\n\n"
        "Your job is to help users set, track, and manage financial savings goals.\n\n"
        "Available tools:\n"
        "  - create_goal_tool: Validate parameters and create or update a FinancialGoal "
        "in a single step. Resolve all values in your reasoning before calling:\n"
        "      goal_name: infer from the item mentioned\n"
        "        (e.g. 'a Mercedes' → 'Mercedes Fund', 'an iPhone' → 'iPhone Fund',\n"
        "        'a house deposit' → 'House Deposit Fund').\n"
        "      target_amount: the numeric savings target.\n"
        "      target_date_iso: resolve any date expression to YYYY-MM-DD yourself\n"
        "        (e.g. 'by the end of 2026' → '2026-12-31',\n"
        "        'by June' → last day of June this or next year,\n"
        "        'in 3 months' → last day of the month 3 months from today,\n"
        "        'next year' → December 31 of next year).\n"
        "      current_amount: amount already saved (default 0).\n"
        "      is_update: true only when modifying an existing goal.\n"
        "    Set a field to null ONLY if it is genuinely absent from the user's message.\n"
        "  - calculate_required_saving_tool: Calculate monthly saving needed for a goal.\n\n"
        "Workflow:\n"
        "  Step 0 — Determine intent before calling any tool:\n"
        "    QUERY (user is asking about / reviewing existing goals, e.g. 'what is my\n"
        "    goal', 'show me my goals', 'how am I tracking'): respond directly from\n"
        "    the Existing goals context — do NOT call create_goal_tool.\n"
        "    CREATE / UPDATE: continue to steps 1–4 below.\n\n"
        "  Steps for CREATE or UPDATE:\n"
        "  1. Extract parameters in your reasoning (resolve date, infer name, extract amount).\n"
        "  2. Call create_goal_tool with the resolved parameters.\n"
        "  3. If status 'missing_fields': ask the user for the missing info in your response.\n"
        "  4. If status 'created' or 'updated': call calculate_required_saving_tool,\n"
        "     then respond with the complete results.\n\n"
        "Context:\n"
        "  - Today: {today}\n"
        "  - Monthly surplus available for savings: {monthly_surplus}\n"
        "  - Existing goals:\n{existing_goals}\n\n"
        "Scope boundary: You are ONLY the goal planning agent. The user message may "
        "mention spending analysis, budgets, or other financial topics — ignore them. "
        "Only create, update, and evaluate savings goals."
    ),
    "react_anomaly_detection": (
        "You are the Anomaly Detection Agent for the SmartFin financial assistant.\n\n"
        "Your job is to scan transactions for suspicious patterns: unusually large "
        "amounts or abnormally high frequency at the same merchant.\n\n"
        "Available tools:\n"
        "  - run_statistical_detection_tool: Run IQR and frequency-based anomaly "
        "detection on transaction data.\n"
        "  - generate_explanation_tool: Generate natural-language explanations for "
        "any anomalies found.\n"
        "  - final_answer: Call this when done.\n\n"
        "Workflow:\n"
        "  1. Call run_statistical_detection_tool with the transaction data.\n"
        "  2. If anomalies are found, call generate_explanation_tool.\n"
        "  3. Call final_answer.\n\n"
        "Note: Statistical detection is the primary method. The tool is deterministic "
        "and reliable. Always run it first to get the ground truth.\n"
        "Context: {transaction_count} transactions available to scan.\n\n"
        "Scope boundary: You are ONLY the anomaly detection agent. The user message may "
        "mention goals, budgets, or spending analysis — ignore them. Only flag and "
        "explain anomalous transactions."
    ),
    "summarise_history": (
        "You are a conversation summariser. Your job is to condense an "
        "earlier portion of a financial-assistant conversation into a "
        "concise summary (2-4 sentences) that preserves:\n"
        "1. Any financial goals the user has set (amount, target date)\n"
        "2. Budget decisions or allocations made\n"
        "3. Key financial data the user provided (income, transactions)\n"
        "4. Any anomalies or health assessment results\n\n"
        "Old messages count: {old_count}\n\n"
        "Transcript:\n{transcript}"
    ),
    "react_health_assessment": (
        "You are the Health Assessment Agent for the SmartFin financial assistant.\n\n"
        "Your job is to evaluate the user's overall financial health by computing "
        "key metrics: debt-to-income ratio, liquid reserves, income concentration "
        "risk, and sustained overspending.\n\n"
        "Available tools:\n"
        "  - compute_health_assessment_tool: Run a complete health assessment on the "
        "user's transaction data. Returns all metrics, rating, and observations.\n"
        "  - final_answer: Call this when done.\n\n"
        "Workflow:\n"
        "  1. Call compute_health_assessment_tool with the transaction data and income.\n"
        "  2. Review the results and call final_answer with a clear summary.\n\n"
        "Context:\n"
        "  - Monthly income: £{income}\n"
        "  - {transaction_count} categorised transactions available\n\n"
        "Scope boundary: You are ONLY the health assessment agent. The user message may "
        "mention goals, budgets, or specific transactions — ignore them. Only assess "
        "overall financial health metrics and provide observations."
    ),
}


# ---------------------------------------------------------------------------
# Inline ChatPromptTemplates for non-ReAct prompts (extraction, classifiers)
# ---------------------------------------------------------------------------

_INLINE_CHAT_PROMPTS: dict[str, ChatPromptTemplate] = {
    "intent_classifier": ChatPromptTemplate.from_messages([
        ("system", "You are a financial intent classifier for SmartFin, a personal finance assistant. "
         "Classify the user's request into exactly one of these categories: "
         "expense_analysis, budget_planning, goal_planning, anomaly_detection, health_assessment, or unknown.\n\n"
         "Examples:\n"
         '- "Analyse my spending" → expense_analysis\n'
         '- "Help me plan a budget" → budget_planning\n'
         '- "I want to save $8000 for an emergency fund" → goal_planning\n'
         '- "Save 1000 by June for an iPhone" → goal_planning\n'
         '- "I need a new laptop, want to set aside 2000" → goal_planning\n'
         '- "Flag any suspicious transactions" → anomaly_detection\n'
         '- "How is my financial health?" → health_assessment\n'
         '- "Write a poem" → unknown\n\n'
         "Key rules:\n"
         "ANY mention of saving money for a future purchase or fund is goal_planning.\n"
         "Prioritize semantic matching, only filter unrelated requests. You can still match the corresponding categories based on the intent though there's a typo.\n\n"
         "User message: {message}"),
    ]),
    "goal_extractor": ChatPromptTemplate.from_messages([
        ("system", "You are a financial goal extractor. Today is {today}. "
         "{context}"
         "Extract the financial goal from the user's message.\n\n"
         "IMPORTANT — Date resolution: Users often express target dates in relative terms. "
         "You MUST resolve these to exact YYYY-MM-DD dates based on today ({today}):\n"
         '  - "end of this month" / "end of the month" / "by the end of this month" → last day of the current month\n'
         '  - "by June" / "by the end of June" → last day of that month (current year if the month is after today, otherwise next year)\n'
         '  - "in N months" → last day of the month N months from now\n'
         '  - "next month" → last day of next month\n'
         '  - "end of this year" → December 31 of the current year\n'
         '  - "by next year" / "end of next year" → December 31 of next year\n'
         '  - "by the end of 2026" / "by 2026" (a specific year) → December 31 of that year\n'
         "If the user provides an exact date (e.g. '2026-06-30'), use it directly.\n\n"
         "Identify: whether it is a goal, a concise goal name (e.g. 'Laptop Fund', "
         "'Emergency Fund'), target amount, target date, and any current savings mentioned. "
         "If the user mentions a specific item or purchase (e.g. 'a Mercedes', 'an iPhone'), "
         "infer a concise goal name from it (e.g. 'Mercedes Fund', 'iPhone Fund'). "
         "If the user wants to adjust, update, modify, or change an existing goal (e.g. "
         "\"update my iPhone goal to 2000\"), set is_update_intent=true and set name to "
         "the goal they want to modify. For updates, only mark fields as missing if they "
         "are required but not provided.\n\n"
         "User message: {user_message}"),
    ]),
    "transaction_extractor": ChatPromptTemplate.from_messages([
        ("system", "Extract all financial transactions explicitly described in the user's message. "
         "Each transaction should have: amount (positive number), merchant or payee name, "
         "short description, and category (one of: food, transport, housing, entertainment, "
         "healthcare, education, shopping, utilities, income, savings, other). "
         "Return an empty list if no transactions are found."),
        ("human", "{message}"),
    ]),
    "expense_categoriser": ChatPromptTemplate.from_messages([
        ("system", "Assign a spending category to each transaction. "
         "Valid categories: {category_values}.\n\n"
         "Transactions:\n{transactions_text}"),
    ]),
    "anomaly_explainer": ChatPromptTemplate.from_messages([
        ("system", "You are an anomaly detection explainer for SmartFin. "
         "Explain why the following transactions are flagged as anomalous, "
         "considering amount, frequency, merchant, and timing patterns.\n\n"
         "Flagged transactions:\n{flagged_transactions_text}"),
    ]),
    "budget_request_extractor": ChatPromptTemplate.from_messages([
        ("system", "You are a budget planning extractor. {context}"
         "Extract the user's budget planning "
         "request details: monthly income, requested categories, and any specific limits. "
         "Supported categories: {supported_categories}\n\n"
         "User message: {last_message}\n"
         "Current known monthly income: {state_income}"),
    ]),
    "health_advisory": ChatPromptTemplate.from_messages([
        ("system", "You are a financial health advisor. Based on the following assessment "
         "results, provide personalised observations and actionable advice.\n\n"
         "Rating: {rating}\n"
         "Debt-to-income ratio: {dti}\n"
         "Liquid reserve months: {reserve_months}\n"
         "Income concentration risk: {concentration_risk}\n"
         "Sustained overspending: {overspending}\n"
         "Monthly income: {monthly_income}\n"
         "Spending trends:\n{trends_text}"),
    ]),
}


def get_react_prompt(name: str, **kwargs) -> str:
    """
    Get a ReAct system prompt and format it with the given variables.
    Uses inline defaults directly (no LangSmith dependency).
    """
    from_string = _REACT_PROMPT_DEFAULTS.get(name, "")
    if not from_string:
        logger.warning("[config] No inline default for '%s'", name)
        return "You are a helpful financial assistant."
    return from_string.format(**kwargs)


def get_prompt(name: str) -> ChatPromptTemplate:
    """
    Get a prompt template by name.
    Returns an inline ChatPromptTemplate (no LangSmith dependency).
    """
    prompt = _INLINE_CHAT_PROMPTS.get(name)
    if prompt is not None:
        return prompt
    raise ValueError(f"Unknown prompt name: '{name}'. Available: {list(_INLINE_CHAT_PROMPTS.keys())}")


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_model_registry() -> dict[str, Any]:
    if _MODEL_REGISTRY_PATH.exists():
        with _MODEL_REGISTRY_PATH.open(encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "schema_version": 1,
        "default_alias": "default",
        "approved_models": {
            "default": {
                "provider": "anthropic",
                "model": _DEFAULT_MODEL,
                "version": "fallback",
                "stage": "prod",
            }
        },
    }


def get_default_model_name() -> str:
    registry = load_model_registry()
    default_alias = registry.get("default_alias", "default")
    approved_models = registry.get("approved_models", {})
    default_entry = approved_models.get(default_alias, {})
    return default_entry.get("model", _DEFAULT_MODEL)


def is_model_approved(model_name: str) -> bool:
    registry = load_model_registry()
    approved_models = registry.get("approved_models", {})
    return any(entry.get("model") == model_name for entry in approved_models.values())


def resolve_model_name(
    requested_model: str | None = None,
    *,
    strict: bool | None = None,
) -> str:
    registry = load_model_registry()
    approved_models = registry.get("approved_models", {})
    strict_mode = _is_truthy(os.getenv("SMARTFIN_ENFORCE_APPROVED_MODELS")) if strict is None else strict

    if not requested_model:
        return get_default_model_name()

    if requested_model in approved_models:
        return approved_models[requested_model].get("model", get_default_model_name())

    if is_model_approved(requested_model):
        return requested_model

    if strict_mode:
        fallback_model = get_default_model_name()
        logger.warning(
            "Requested model '%s' is not approved; falling back to '%s'.",
            requested_model,
            fallback_model,
        )
        return fallback_model

    return requested_model


def get_llm(alias: str | None = None, **kwargs):
    """
    Return a ChatOpenAI client routed through the LiteLLM gateway.

    Uses resolve_model_name(alias) to look up the model from the registry,
    then connects to the gateway via LITELLM_BASE_URL / LITELLM_VIRTUAL_KEY.
    Pass extra kwargs (timeout, temperature, etc.) as needed per call site.
    """
    import os
    from langchain_openai import ChatOpenAI

    model_name = resolve_model_name(alias)
    return ChatOpenAI(
        model=model_name,
        base_url=os.environ.get("LITELLM_BASE_URL", "http://gateway:4000/v1"),
        api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "placeholder"),
        **kwargs,
    )


def get_monitoring_settings() -> dict[str, Any]:
    return {
        "langsmith_tracing": _is_truthy(os.getenv("LANGSMITH_TRACING")),
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "smartfin"),
        "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "log_format": os.getenv("SMARTFIN_LOG_FORMAT", "plain").lower(),
    }


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _EXTRA_LOG_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, force: bool = False) -> logging.Logger:
    settings = get_monitoring_settings()
    level_name = settings["log_level"]
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if root_logger.handlers and not force:
        root_logger.setLevel(level)
        return root_logger

    handler = logging.StreamHandler()
    if settings["log_format"] == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

    root_logger.handlers = [handler]
    root_logger.setLevel(level)
    return root_logger
