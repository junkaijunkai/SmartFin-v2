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
from typing import Annotated, Any, Optional
from pydantic import Field

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from app.orchestrator.state_view import AgentStateView
from app.agents.react_utils import run_react_loop
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
    _state_income = monthly_income  # capture before tool parameter shadows outer variable
    tool_ctx: dict[str, Any] = {}

    @tool
    def extract_budget_request_tool(
        monthly_income: Annotated[
            Optional[float],
            Field(description=(
                "The user's gross monthly income. Extract from the message "
                "(e.g. 'I earn $5000/month', 'my salary is £3000') or use the value "
                "already shown in the financial context above. "
                "Set to null only if income is genuinely unknown from both sources."
            )),
        ],
        categories: Annotated[
            list[str],
            Field(description=(
                "Spending categories the user wants to budget for "
                "(e.g. ['food', 'transport', 'entertainment']). "
                "Use an empty list to include all available categories. "
                "Valid values: food, transport, housing, entertainment, healthcare, "
                "education, shopping, utilities, savings, other."
            )),
        ] = [],
    ) -> str:
        """
        Record the budget parameters you have identified from the user's message and context.

        Call this once after reading the user's request and the financial context provided.
        Extract monthly_income and any category preferences in your own reasoning first,
        then call this tool to commit the result. Do not use this tool to re-parse the message.

        If monthly_income is null (not found in message or context), this tool returns a
        signal that clarification is needed. You must then call final_answer to ask the user
        for their monthly income before generating any allocations.

        Returns the recorded parameters and a 'needs_clarification' flag.
        """
        effective_income = monthly_income if monthly_income is not None else _state_income
        result = {
            "monthly_income": effective_income,
            "categories_requested": categories,
            "needs_clarification": effective_income is None,
        }
        tool_ctx["budget_request"] = result
        if result["needs_clarification"]:
            return (
                "Monthly income is missing from both the user message and context. "
                "You must ask the user for their monthly income before proceeding."
            )
        return json.dumps(result, default=str)

    @tool
    def generate_allocations_tool(
        monthly_income_arg: Annotated[
            float,
            Field(description="The user's gross monthly income used as the allocation baseline."),
        ],
        category_avg_data: Annotated[
            dict[str, float],
            Field(description="Dict mapping each spending category name to its average monthly spend."),
        ],
        category_trend_data: Annotated[
            dict[str, str],
            Field(description=(
                "Dict mapping each category to its spending trend direction. "
                "Valid values: 'rising', 'stable', 'volatile', 'fixed'."
            )),
        ],
        existing_budget_data: Annotated[
            Optional[dict[str, float]],
            Field(description=(
                "Optional dict of existing budget allocations (category → allocated amount). "
                "Pass null when creating a fresh budget with no prior allocations."
            )),
        ] = None,
    ) -> str:
        """
        Generate recommended budget allocations for each spending category.

        Call this after extract_budget_request_tool confirms monthly income is available.
        Use the spending averages and trends from the financial context provided in the
        system prompt. Do not call this if needs_clarification was true.

        Returns a dict mapping category name to the recommended monthly allocation amount.
        If a category has a 'rising' trend, its allocation will be adjusted upward.
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
        transactions_data: Annotated[
            list[dict],
            Field(description=(
                "List of categorised transaction objects. Each must have 'amount' (float) "
                "and 'category' (str) fields. Pass all available categorised transactions."
            )),
        ],
    ) -> str:
        """
        Compute total actual spending per category for the current month.

        Call this after generate_allocations_tool to get real spending figures
        to compare against the budget targets. Use the categorised transactions
        available in context. Do not call this before allocations are generated.

        Returns a dict mapping category name to total amount spent this month.
        """
        result = calculate_monthly_spending(transactions_data)
        tool_ctx["actual_spending"] = result
        return json.dumps(result, default=str)

    @tool
    def evaluate_progress_tool(
        allocations: Annotated[
            dict[str, float],
            Field(description="Budget allocations from generate_allocations_tool (category → amount)."),
        ],
        spending: Annotated[
            dict[str, float],
            Field(description="Actual spending from calculate_spending_tool (category → amount spent)."),
        ],
        day_of_month: Annotated[
            int,
            Field(description="Current day of the month (1–31). Use the value from the context."),
        ],
        total_days: Annotated[
            int,
            Field(description="Total days in the current month (28–31). Use the value from context."),
        ],
    ) -> str:
        """
        Evaluate how actual spending compares against budget allocations mid-month.

        Call this after both generate_allocations_tool and calculate_spending_tool
        have returned. Use the day_of_month and total_days values from the financial
        context. Do not call this before allocations and spending data are available.

        Returns per-category progress data including pacing status and projected overspend.
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
        progress_data: Annotated[
            dict[str, Any],
            Field(description="The progress dict returned by evaluate_progress_tool."),
        ],
    ) -> str:
        """
        Generate budget warnings for categories at risk of exceeding their allocation.

        Call this after evaluate_progress_tool has returned. Pass the progress dict
        exactly as returned by evaluate_progress_tool. Do not call this before
        budget progress has been evaluated.

        Returns a list of warning objects, each with 'category', 'severity'
        ('low', 'medium', 'high'), and 'message' describing the risk.
        """
        result = generate_budget_warnings(progress_data)
        tool_ctx["warnings"] = result
        return json.dumps(result, default=str)

    @tool
    def validate_budget_tool(
        allocations_json: Annotated[
            list[dict],
            Field(description=(
                "List of BudgetAllocation dicts from generate_allocations_tool. "
                "Each must have 'category', 'allocated_amount', 'spent_amount', "
                "'period_start', and 'period_end' fields."
            )),
        ],
        progress_json: Annotated[
            dict[str, Any],
            Field(description="Progress dict from evaluate_progress_tool."),
        ],
        warnings_json: Annotated[
            list[dict],
            Field(description="Warnings list from generate_warnings_tool."),
        ],
        summary_text: Annotated[
            str,
            Field(description="A human-readable summary of the budget plan (1–3 sentences)."),
        ],
        request_json: Annotated[
            dict[str, Any],
            Field(description="Budget request dict from extract_budget_request_tool."),
        ],
    ) -> str:
        """
        Validate the structural correctness of the complete budget output.

        Call this only when all preceding tools have been called and their results
        are available: allocations, progress, warnings, summary, and request.
        Do not call this before the full pipeline has completed.
        If validation fails, fix the reported errors and call this tool again.

        Returns 'Budget output is valid.' on success, or a description of errors to fix.
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

    # Build recent conversation history for multi-turn context
    history_msgs = [
        m for m in (messages_list or [])
        if hasattr(m, 'type') and m.type in ('human', 'ai') and getattr(m, 'content', '')
    ][:-1]  # Exclude the latest message (already in user_message)

    response, _ = run_react_loop(
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
        history=history_msgs[-6:] if history_msgs else None,
    )

    # ------------------------------------------------------------------
    # Post-processing: build state update from tool_ctx
    # ------------------------------------------------------------------
    budget_request = tool_ctx.get("budget_request", {})

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

    default_text = (
        f"Budget planning completed for {len(raw_allocations)} categories. "
        f"{len(warnings)} warning(s) generated."
    )
    summary = response.content or default_text

    return {
        "budget_allocations": allocation_list,
        "budget_progress": progress,
        "budget_warnings": warnings,
        "budget_summary": summary,
        "budget_request": budget_request,
        "messages": [AIMessage(content=summary)],
    }
