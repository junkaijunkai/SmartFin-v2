"""
Expense Analysis agent — LangGraph node entry point with ReAct loop.

Transformed from a single-pass function into a ReAct agent that reasons about
the available data, calls tools to categorise transactions and compute trends,
and produces a structured final answer.

Responsibilities:
  1. Pre-process: extract transactions from user message, merge with state.
  2. ReAct loop: LLM reasons → calls categorise → observes → calls compute
     trends → observes → calls final_answer.
  3. Post-process: read tool_ctx and build AppState updates + HITL payload.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from app.state import SpendingTrend, Transaction
from app.orchestrator.state_view import AgentStateView
from app.agents.react_utils import run_react_loop, final_answer as shared_final_answer
from app.agents.expense_analysis.categoriser import categorise_transactions
from app.agents.expense_analysis.analyser import compute_spending_trends
from app.agents.expense_analysis.extractor import extract_transaction_from_message
from app.config import resolve_model_name, get_react_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_trend(deviation_pct: float | None) -> str:
    if deviation_pct is None:
        return "fixed"
    if deviation_pct >= 10:
        return "rising"
    if deviation_pct <= -10:
        return "volatile"
    return "stable"


def _serialise_transactions(txns: list[Transaction]) -> list[dict]:
    """Convert Transaction objects to plain dicts for tool I/O."""
    return [t.model_dump(mode="json") for t in txns]


def _deserialise_transactions(
    raw: list[dict],
) -> list[Transaction]:
    """Convert plain dicts back to Transaction objects."""
    return [Transaction(**t) for t in raw]


# ---------------------------------------------------------------------------
# ReAct agent node
# ---------------------------------------------------------------------------


def run(view: AgentStateView, config: RunnableConfig | None = None) -> dict:
    """
    LangGraph node function with ReAct loop for expense analysis.
    """
    # ------------------------------------------------------------------
    # Pre-processing (deterministic data plumbing)
    # ------------------------------------------------------------------
    new_transactions = list(view.get("transactions") or [])
    existing_categorised = list(view.get("categorised_transactions") or [])

    messages = view.get("messages") or []
    if messages and isinstance(messages[-1], HumanMessage):
        extracted_list = extract_transaction_from_message(
            messages[-1].content, view.get("current_date")
        )
        if extracted_list:
            known_ids = {t.id for t in new_transactions} | {
                t.id for t in existing_categorised
            }
            for extracted in extracted_list:
                if extracted.id not in known_ids:
                    new_transactions.append(extracted)
                    known_ids.add(extracted.id)

    existing_ids = {t.id for t in existing_categorised}
    to_categorise = [t for t in new_transactions if t.id not in existing_ids]

    # -- Early return if there's nothing to process at all --
    if not to_categorise and not existing_categorised:
        return _build_result([], [], llm_succeeded=True)

    # -- If nothing NEW to categorise, skip the ReAct loop entirely --
    if not to_categorise and existing_categorised:
        trends = compute_spending_trends(existing_categorised)
        return _build_result(existing_categorised, trends, llm_succeeded=True)

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------
    tool_ctx: dict[str, Any] = {}

    # --- Tools (wrapping existing helpers) ---

    @tool
    def categorise_transactions_tool(
        transactions: list[dict],
    ) -> str:
        """
        Categorise a list of raw transactions by assigning a spending category
        (food, transport, housing, entertainment, etc.) to each one.

        The input should be a JSON array of transaction objects, each with
        'id', 'amount', 'description', and 'merchant' fields.  Returns the
        same list with the 'category' field populated.
        """
        txns = _deserialise_transactions(transactions)
        categorised, llm_ok = categorise_transactions(txns)
        merged = existing_categorised + categorised
        tool_ctx["categorised"] = merged
        tool_ctx["categorise_llm_ok"] = llm_ok
        return json.dumps(
            _serialise_transactions(categorised), default=str
        )

    @tool
    def compute_spending_trends_tool(
        categorised: list[dict],
    ) -> str:
        """
        Compute 30-day spending trends from a list of categorised transactions.

        The input should be a JSON array of categorised transaction objects
        (each with 'category', 'amount', 'date' fields).  Returns a JSON
        array of trend objects, each with 'category', 'current_period_total',
        'previous_period_total', and 'deviation_pct'.
        """
        txns = _deserialise_transactions(categorised)
        trends = compute_spending_trends(txns)
        tool_ctx["trends"] = trends
        return json.dumps(
            [t.model_dump(mode="json") for t in trends], default=str
        )

    # --- Build LLM with tools ---
    model_name = resolve_model_name()
    llm = ChatAnthropic(
        model=model_name, timeout=30
    ).bind_tools([categorise_transactions_tool, compute_spending_trends_tool, shared_final_answer])

    tools_map: dict[str, Any] = {
        "categorise_transactions_tool": categorise_transactions_tool,
        "compute_spending_trends_tool": compute_spending_trends_tool,
    }

    # --- System prompt (from LangSmith) ---
    system = get_react_prompt(
        "react_expense_analysis",
        transaction_count=len(to_categorise),
        existing_count=len(existing_categorised),
    )

    # --- User message ---
    user_msg = messages[-1].content if messages else ""

    # --- Run ReAct loop ---
    run_react_loop(
        llm=llm,
        tools=tools_map,
        system_prompt=system,
        user_message=(
            f"User request: {user_msg}\n\n"
            f"There are {len(to_categorise)} new transactions to process "
            f"and {len(existing_categorised)} already categorised.\n"
            f"New transactions: {json.dumps(_serialise_transactions(to_categorise), default=str)}"
        ),
        tool_ctx=tool_ctx,
    )

    # ------------------------------------------------------------------
    # Post-processing: build state update from tool_ctx
    # ------------------------------------------------------------------
    categorised: list[Transaction] = tool_ctx.get(
        "categorised", existing_categorised + to_categorise
    )
    trends: list[SpendingTrend] = tool_ctx.get(
        "trends", compute_spending_trends(categorised)
    )
    llm_ok = tool_ctx.get("categorise_llm_ok", True)

    return _build_result(categorised, trends, llm_ok, tool_ctx)


# ---------------------------------------------------------------------------
# State update builder
# ---------------------------------------------------------------------------


def _build_result(
    categorised: list[Transaction],
    trends: list[SpendingTrend],
    llm_succeeded: bool,
    tool_ctx: dict | None = None,
) -> dict:
    """Build the HITL confirmation and state update from tool results."""
    trend_lines: list[str] = []
    for t in trends:
        if t.deviation_pct is None:
            trend_lines.append(
                f"  {t.category.value:<15} £{t.current_period_total:>8.2f}"
                f"  No data for prev period"
            )
        else:
            sign = "+" if t.deviation_pct >= 0 else "-"
            trend_lines.append(
                f"  {t.category.value:<15} £{t.current_period_total:>8.2f}"
                f"  ({sign}{t.deviation_pct:.1f}% vs prev period)"
            )

    # Use ReAct summary if available, otherwise build default
    summary_text = (tool_ctx or {}).get(
        "summary",
        (
            f"Categorised {len(categorised)} transactions across "
            f"{len(trends)} spending categories."
        ),
    )
    needs_hitl = (tool_ctx or {}).get("needs_hitl", True)

    pending_confirmation = None
    if needs_hitl:
        hitl_summary = (tool_ctx or {}).get(
            "hitl_summary", summary_text
        )
        hitl_details = (tool_ctx or {}).get(
            "hitl_details", trend_lines
        )
        pending_confirmation = {
            "action": "approve_expense_analysis",
            "agent": "expense_analysis",
            "summary": hitl_summary,
            "details": hitl_details,
            "categorisation_confidence": (
                "llm" if llm_succeeded else "fallback_keywords"
            ),
        }

    category_monthly_avg = {
        trend.category.value: trend.current_period_total for trend in trends
    }
    category_trends = {
        trend.category.value: _classify_trend(trend.deviation_pct)
        for trend in trends
    }

    return {
        "categorised_transactions": categorised,
        "spending_trends": trends,
        "expense_analysis": {
            "category_monthly_avg": category_monthly_avg,
            "category_trends": category_trends,
        },
        "pending_confirmation": pending_confirmation,
        "messages": [AIMessage(content=summary_text)],
    }
