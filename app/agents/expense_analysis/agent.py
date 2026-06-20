"""
Expense Analysis agent — LangGraph node entry point with ReAct loop.

Transformed from a single-pass function into a ReAct agent that reasons about
the available data, calls tools to categorise transactions and compute trends,
and produces a natural-language answer.

Responsibilities:
  1. Pre-process: extract transactions from user message, merge with state.
  2. ReAct loop: LLM reasons → calls categorise → observes → calls compute
     trends → observes → produces end_turn text response.
  3. Post-process: read tool_ctx and build AppState updates + HITL payload.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any
from pydantic import Field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from app.state import SpendingTrend, Transaction
from app.orchestrator.state_view import AgentStateView
from app.agents.react_utils import run_react_loop
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
        transactions: Annotated[
            list[dict],
            Field(description=(
                "List of raw, uncategorised transaction objects. Each object must include "
                "'id' (str), 'amount' (float), 'description' (str), 'merchant' (str), "
                "and 'date' (YYYY-MM-DD). Pass ONLY transactions that have not yet been "
                "categorised — do not re-pass items already returned by a prior call."
            )),
        ],
    ) -> str:
        """
        Assign a spending category to each raw transaction and return the categorised list.

        Call this once per agent turn for all new transactions that lack a 'category' field.
        Do not pass transactions already present in the categorised state — those are merged
        in automatically. Valid categories: food, transport, housing, entertainment,
        healthcare, education, shopping, utilities, income, savings, other.

        Returns the newly categorised transactions with 'category' populated. The stored
        result merges with previously categorised transactions held in state. If the LLM
        categorisation fails, a keyword-based fallback assigns best-guess categories.
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
        categorised: Annotated[
            list[dict],
            Field(description=(
                "Complete list of categorised transactions to analyse — both newly "
                "categorised and those already in state. Each object must have 'category' "
                "(str), 'amount' (float), and 'date' (YYYY-MM-DD). Do not pass raw "
                "uncategorised transactions; call categorise_transactions_tool first."
            )),
        ],
    ) -> str:
        """
        Compute 30-day spending trends from the full set of categorised transactions.

        Call this after categorise_transactions_tool has returned. Pass the complete
        merged list of categorised transactions (new + existing from state), not just
        the newly categorised ones. Do not call this if there are no categorised
        transactions available.

        Returns one trend object per spending category with: 'category',
        'current_period_total', 'previous_period_total', and 'deviation_pct'
        (positive = spending rose vs. prior 30 days; negative = fell; null = no prior data).
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
    ).bind_tools([categorise_transactions_tool, compute_spending_trends_tool])

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

    # --- Build recent conversation history for multi-turn context ---
    history_msgs = [
        m for m in (messages or [])
        if hasattr(m, 'type') and m.type in ('human', 'ai') and getattr(m, 'content', '')
    ][:-1]  # Exclude the latest message (already in user_message)

    # --- Run ReAct loop ---
    response, _ = run_react_loop(
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
        history=history_msgs[-6:] if history_msgs else None,
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

    default_summary = (
        f"Categorised {len(categorised)} transactions across "
        f"{len(trends)} spending categories."
    )
    summary_text = response.content or default_summary
    return _build_result(categorised, trends, llm_ok, tool_ctx, summary_text)


# ---------------------------------------------------------------------------
# State update builder
# ---------------------------------------------------------------------------


def _build_result(
    categorised: list[Transaction],
    trends: list[SpendingTrend],
    llm_succeeded: bool,
    tool_ctx: dict | None = None,
    summary_text: str = "",
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

    # HITL fires whenever categorisation actually ran
    needs_hitl = bool((tool_ctx or {}).get("categorised"))

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
