"""
Financial Health and Risk Assessment agent — LangGraph node entry point.

Directly computes health metrics without ReAct overhead, since the assessment
is entirely deterministic (compute + LLM advisory bundled in assess_health).

Design:
  - assess_health() in assessor.py handles both deterministic metrics and
    LLM advisory generation (with silent fallback).
  - This node is a thin wrapper that reads state, calls assess_health(),
    and formats the response.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from app.orchestrator.state_view import AgentStateView
from app.agents.health_assessment.assessor import assess_health

logger = logging.getLogger(__name__)


def run(view: AgentStateView) -> dict:
    """
    LangGraph node entry point for health assessment.
    """
    # ------------------------------------------------------------------
    # Read data from state
    # ------------------------------------------------------------------
    transactions = (
        view.get("categorised_transactions")
        or view.get("transactions")
        or []
    )
    monthly_income = view.get("monthly_income") or 0.0
    spending_trends = view.get("spending_trends") or []
    existing_alerts = list(view.get("alerts") or [])

    if not transactions or monthly_income <= 0:
        return {
            "health_summary": None,
            "alerts": existing_alerts,
            "messages": [
                AIMessage(
                    content="I need transaction data and your monthly income "
                            "to assess your financial health."
                )
            ],
        }

    # Resolve reference date from state (for rolling 30-day windows)
    current_date_str = view.get("current_date")
    reference_date = None
    if current_date_str:
        try:
            reference_date = datetime.fromisoformat(current_date_str).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            reference_date = None

    # ------------------------------------------------------------------
    # Compute health assessment directly
    # ------------------------------------------------------------------
    health_summary, new_alerts = assess_health(
        transactions=transactions,
        monthly_income=monthly_income,
        spending_trends=spending_trends,
        reference_date=reference_date,
    )
    merged_alerts = existing_alerts + new_alerts

    # Build response text
    rating = health_summary.rating.value.upper()
    emoji = {"GOOD": "\U0001f7e2", "FAIR": "\U0001f7e1", "POOR": "\U0001f534"}.get(
        rating, ""
    )
    summary_text = (
        f"{emoji} **Financial Health: {rating}**\n\n"
        f"- Debt-to-income ratio: {health_summary.debt_to_income_ratio:.0%}\n"
        f"- Liquid reserves: {health_summary.liquid_reserve_months:.1f} months\n"
        + (
            "- Income concentration risk detected\n"
            if health_summary.income_concentration_risk
            else ""
        )
        + (
            "- Sustained overspending detected\n"
            if health_summary.sustained_overspending
            else ""
        )
        + (
            "\n**Advisory:**\n"
            + "\n".join(f"- {o}" for o in health_summary.observations)
            if health_summary.observations
            else ""
        )
    )

    return {
        "health_summary": health_summary,
        "alerts": merged_alerts,
        "messages": [AIMessage(content=summary_text)],
    }
