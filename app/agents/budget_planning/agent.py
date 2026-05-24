"""
Budget Planning agent — LangGraph node entry point with ReAct loop.

Transformed from a single-pass function into a ReAct agent. The LLM reasons
about the user's budget request, calls tools to extract parameters, generate
allocations, evaluate progress, and validate output.

Responsibilities:
  1. Read shared state for expense analysis data and user preferences.
  2. ReAct loop: LLM extracts budget intent → generates allocations →
     evaluates progress → validates → calls final_answer.
  3. Write structured budget data back to AppState.
"""

from __future__ import annotations

import calendar
import json
import logging
from datetime import date, datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from app.orchestrator.state_view import AgentStateView
from app.agents.react_utils import run_react_loop, final_answer as shared_final_answer
from app.agents.budget_planning.extractor import extract_budget_request as _extract
from app.agents.budget_planning.planner import (
    generate_budget_allocations,
    calculate_monthly_spending,
    evaluate_budget_progress,
    generate_budget_warnings,
)
from app.config import resolve_model_name, get_react_prompt
from app.guardrails.output_validator import validate_budget_output
from app.state import BudgetAllocation, TransactionCategory

logger = logging.getLogger(__name__)


def budget_planning_node(view: AgentStateView) -> dict:
    """
    LangGraph node entry point with ReAct loop for budget planning.
    """
    # ------------------------------------------------------------------
    # Read context from state
    # ------------------------------------------------------------------
    monthly_income = view.get("monthly_income")
    categorised = view.get("categorised_transactions") or []
    if not categorised:
        categorised = view.get("transactions", default=[])

    expense_analysis = view.get("expense_analysis", default={}) or {}
    category_monthly_avg = (
        expense_analysis.get("category_monthly_avg", {}) or {}
    )
    category_trends = expense_analysis.get("category_trends", {}) or {}

    raw_existing = view.get("budget_allocations") or []
    existing_budget: dict[str, float] = {}
    for alloc in raw_existing:
        if isinstance(alloc, BudgetAllocation):
            existing_budget[alloc.category.value] = alloc.allocated_amount

    current_date_str = view.get("current_date")
    if current_date_str:
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    else:
        current_date = datetime.today()
    current_day = current_date.day
    days_in_month = calendar.monthrange(
        current_date.year, current_date.month
    )[1]

    messages_list = view.get("messages") or []
    user_msg = messages_list[-1].content if messages_list else ""

    extractor_context: str | None = None

    # Build a compact summary of categories for the system prompt context
    category_lines = []
    for cat, avg in category_monthly_avg.items():
        trend = category_trends.get(cat, "stable")
        ex_budget = existing_budget.get(cat)
        ex_str = f", existing_budget={ex_budget:.2f}" if ex_budget else ""
        category_lines.append(
            f"  {cat}: avg_spend={avg:.2f}, trend={trend}{ex_str}"
        )
    categories_context = "\n".join(category_lines) or "  (no spending data)"

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------
    tool_ctx: dict[str, Any] = {}

    @tool
    def extract_budget_request_tool(
        user_message: str,
        provided_monthly_income: float | None = None,
    ) -> str:
        """
        Analyse the user's natural-language budget request and extract
        structured information: monthly income, categories of interest, and
        whether more information is needed.

        Provide the user's original message and any monthly income you have
        (or None if unknown).  Returns structured budget request fields.
        """
        extract_state = {
            "messages": [AIMessage(content=user_message)],
            "monthly_income": provided_monthly_income,
        }
        result = _extract(extract_state, context=extractor_context)
        tool_ctx["budget_request"] = result

        if result.get("needs_clarification"):
            return (
                "Monthly income is missing. The user needs to provide their "
                "monthly income before I can create a budget plan."
            )

        return json.dumps(result, default=str)

    @tool
    def generate_allocations_tool(
        monthly_income_arg: float,
        category_avg_data: dict[str, float],
        category_trend_data: dict[str, str],
        existing_budget_data: dict[str, float] | None = None,
    ) -> str:
        """
        Generate budget allocations for each spending category based on income,
        historical spending averages, and spending trends.

        Parameters:
          monthly_income_arg: The user's monthly income.
          category_avg_data: Dict mapping category name to average monthly spend.
          category_trend_data: Dict mapping category name to trend direction
            ('rising', 'stable', 'volatile', 'fixed').
          existing_budget_data: Optional dict of existing budget allocations.
        """
        raw = generate_budget_allocations(
            monthly_income=monthly_income_arg,
            category_monthly_avg=category_avg_data,
            category_trends=category_trend_data,
            existing_budget=existing_budget_data or {},
        )
        tool_ctx["raw_allocations"] = raw
        return json.dumps(raw, default=str)

    @tool
    def calculate_spending_tool(
        transactions_data: list[dict],
    ) -> str:
        """
        Calculate actual monthly spending from a list of categorised transactions.

        Input should be a JSON array of transaction objects with 'amount',
        'category' fields.  Returns a dict of category → total spent.
        """
        result = calculate_monthly_spending(transactions_data)
        tool_ctx["actual_spending"] = result
        return json.dumps(result, default=str)

    @tool
    def evaluate_progress_tool(
        allocations: dict[str, float],
        spending: dict[str, float],
        day_of_month: int,
        total_days: int,
    ) -> str:
        """
        Evaluate how actual spending compares against budget allocations.

        Parameters:
          allocations: Dict of category → budgeted amount.
          spending: Dict of category → actual amount spent.
          day_of_month: Current day number (1-31).
          total_days: Total days in the current month (28-31).
        """
        result = evaluate_budget_progress(
            budget_allocations=allocations,
            actual_spending=spending,
            current_day=day_of_month,
            days_in_month=total_days,
        )
        tool_ctx["progress"] = result
        return json.dumps(result, default=str)

    @tool
    def generate_warnings_tool(
        progress_data: dict[str, Any],
    ) -> str:
        """
        Generate budget warnings based on progress data.

        Input is the progress dict from evaluate_progress_tool.
        Returns a list of warning dicts with 'category', 'severity', 'message'.
        """
        result = generate_budget_warnings(progress_data)
        tool_ctx["warnings"] = result
        return json.dumps(result, default=str)

    @tool
    def validate_budget_tool(
        allocations_json: list[dict],
        progress_json: dict[str, Any],
        warnings_json: list[dict],
        summary_text: str,
        request_json: dict[str, Any],
    ) -> str:
        """
        Validate the complete budget output for structural correctness.

        Returns a validation result dict with 'valid' (bool) and 'errors' (list).
        Only call this when you have all the budget data ready.
        """
        candidate = {
            "budget_allocations": [
                BudgetAllocation(**a) for a in allocations_json
            ],
            "budget_progress": progress_json,
            "budget_warnings": warnings_json,
            "budget_summary": summary_text,
            "budget_request": request_json,
        }
        result = validate_budget_output(candidate)
        tool_ctx["validation"] = result

        if result["valid"]:
            tool_ctx["budget_allocations_obj"] = candidate["budget_allocations"]
            tool_ctx["budget_progress_obj"] = progress_json
            tool_ctx["budget_warnings_obj"] = warnings_json
            return "Budget output is valid."

        return (
            f"Validation found errors: {result['errors']}. "
            "Please fix the issues and re-validate."
        )

    # --- Build LLM with all tools ---
    model_name = resolve_model_name()
    llm = ChatAnthropic(
        model=model_name, timeout=30
    ).bind_tools([
        extract_budget_request_tool,
        generate_allocations_tool,
        calculate_spending_tool,
        evaluate_progress_tool,
        generate_warnings_tool,
        validate_budget_tool,
        shared_final_answer,
    ])

    tools_map: dict[str, Any] = {
        "extract_budget_request_tool": extract_budget_request_tool,
        "generate_allocations_tool": generate_allocations_tool,
        "calculate_spending_tool": calculate_spending_tool,
        "evaluate_progress_tool": evaluate_progress_tool,
        "generate_warnings_tool": generate_warnings_tool,
        "validate_budget_tool": validate_budget_tool,
    }

    # --- System prompt (context about available data) ---
    system_text = get_react_prompt(
        "react_budget_planning",
        monthly_income=monthly_income or "unknown",
        categories=categories_context,
        current_month=current_date.strftime("%B %Y"),
        current_day=current_day,
        days_in_month=days_in_month,
    )

    # Append relevant user history from long-term memory, if available.
    memory_ctx = view.get("memory_context") or ""
    if memory_ctx:
        system_text += f"\n\n## User History\n{memory_ctx}"

    react_msg = user_msg
    if extractor_context:
        react_msg = (
            f"[Continuing budget conversation. Context: {extractor_context}]\n\n"
            f"{user_msg}"
        )

    run_react_loop(
        llm=llm,
        tools=tools_map,
        system_prompt=system_text,
        user_message=(
            f"User budget request: {react_msg}\n\n"
            f"Available context:\n"
            f"- Monthly income: {monthly_income or 'unknown'}\n"
            f"- Current date: {current_date_str or datetime.today().strftime('%Y-%m-%d')}\n"
            f"- {len(categorised)} categorised transactions available\n"
            f"- {len(existing_budget)} existing budget allocations\n"
            f"- Spending data by category:\n{categories_context}"
        ),
        tool_ctx=tool_ctx,
    )

    # ------------------------------------------------------------------
    # Post-processing: build state update from tool_ctx
    # ------------------------------------------------------------------
    budget_request = tool_ctx.get("budget_request", {})

    # Handle clarification path (missing monthly income)
    if budget_request.get("needs_clarification"):
        return {
            "budget_request": budget_request,
            "budget_summary": (
                "More information is needed before generating a budget plan."
            ),
            "budget_warnings": [],
            "budget_progress": {},
            "output_validation_result": {
                "valid": True,
                "errors": [],
                "sanitized_output": None,
            },
            "pending_confirmation": {
                "action": "clarify_budget_planning",
                "agent": "budget_planning",
                "summary": "Monthly income is required to generate a budget plan.",
                "details": [
                    "Please provide your monthly income so I can calculate "
                    "budget allocations.",
                ],
            },
        }

    # Build BudgetAllocation objects from tool results
    raw_allocations = tool_ctx.get("raw_allocations", {})
    actual_spending = tool_ctx.get(
        "actual_spending", calculate_monthly_spending(categorised)
    )
    progress = tool_ctx.get(
        "progress",
        evaluate_budget_progress(
            raw_allocations,
            actual_spending,
            current_day,
            days_in_month,
        ),
    )
    warnings = tool_ctx.get(
        "warnings",
        generate_budget_warnings(progress),
    )

    period_start = date(current_date.year, current_date.month, 1)
    period_end = date(
        current_date.year, current_date.month, days_in_month
    )

    allocation_list: list[BudgetAllocation] = []
    for cat, amount in raw_allocations.items():
        if cat == TransactionCategory.INCOME.value:
            continue
        try:
            category_enum = TransactionCategory(cat)
        except ValueError:
            continue
        allocation_list.append(
            BudgetAllocation(
                category=category_enum,
                allocated_amount=amount,
                spent_amount=actual_spending.get(cat, 0.0),
                period_start=period_start,
                period_end=period_end,
            )
        )

    summary = tool_ctx.get(
        "summary",
        (
            f"Budget planning completed for {len(raw_allocations)} categories. "
            f"{len(warnings)} warning(s) generated."
        ),
    )

    return {
        "budget_allocations": allocation_list,
        "budget_progress": progress,
        "budget_warnings": warnings,
        "budget_summary": summary,
        "budget_request": budget_request,
        "messages": [AIMessage(content=summary)],
    }
