"""Memory writer — persists agent task outputs as natural-language embeddings."""

from __future__ import annotations

import json
import logging
from datetime import date

from app.memory.embeddings import embed
from app.memory.vector_store import upsert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_task_memory(agent_name: str, state: dict) -> None:
    """Write one memory record per completed task to the vector store.

    Reads memory fields from AGENT_SCOPES, serialises each field to natural
    language, embeds the text, and upserts into the pgvector store.

    Failures are caught and logged as warnings — this function is non-critical
    and must never raise.
    """
    try:
        # Import here to avoid circular imports (orchestrator → memory → orchestrator)
        from app.orchestrator.state_view import AGENT_SCOPES

        scope = AGENT_SCOPES.get(agent_name, {})
        memory_fields: set[str] = scope.get("memory", set())

        if not memory_fields:
            return

        if "goals" in memory_fields:
            goals = state.get("goals") or []
            _write_goals(goals, agent_name)

        if "budget_allocations" in memory_fields:
            allocs = state.get("budget_allocations") or []
            income = state.get("monthly_income")
            _write_budget(allocs, income, agent_name)

    except Exception as exc:
        logger.warning("[memory.writer] write_task_memory failed for %s: %s", agent_name, exc)


# ---------------------------------------------------------------------------
# Private helpers — goal persistence
# ---------------------------------------------------------------------------


def _write_goals(goals: list, agent_name: str) -> None:
    """Write one vector-store record per FinancialGoal."""
    if not goals:
        return

    for goal in goals:
        try:
            text = _goal_to_text(goal)
            description = (
                f"Goal '{goal.name}': ${goal.current_amount:,.2f}/${goal.target_amount:,.2f}"
                f" by {goal.target_date},"
                f" {'on track' if goal.on_track else 'behind'}"
            )
            record_id = f"{agent_name}/{_safe_slug(goal.name)}"
            embedding = embed(text)
            upsert(record_id, "goal", text, description, embedding)
        except Exception as exc:
            logger.warning("[memory.writer] Failed to write goal '%s': %s", getattr(goal, "name", "?"), exc)


def _goal_to_text(goal) -> str:
    """Serialise a FinancialGoal to a natural-language sentence."""
    try:
        pct = (goal.current_amount / goal.target_amount * 100) if goal.target_amount else 0.0
        status = "on track" if goal.on_track else "behind schedule"
        return (
            f"Savings goal '{goal.name}': targeting ${goal.target_amount:,.2f}"
            f" by {goal.target_date}."
            f" Currently saved ${goal.current_amount:,.2f} ({pct:.1f}%)."
            f" Requires ${goal.required_monthly_saving:,.2f}/month."
            f" Status: {status}."
        )
    except Exception as exc:
        logger.warning("[memory.writer] Goal text serialisation failed, using fallback: %s", exc)
        # Fallback: JSON key fields
        return json.dumps(
            {
                "name": getattr(goal, "name", None),
                "target_amount": getattr(goal, "target_amount", None),
                "current_amount": getattr(goal, "current_amount", None),
                "target_date": str(getattr(goal, "target_date", None)),
                "on_track": getattr(goal, "on_track", None),
            },
            default=str,
        )


# ---------------------------------------------------------------------------
# Private helpers — budget persistence
# ---------------------------------------------------------------------------


def _write_budget(allocs: list, income: float | None, agent_name: str) -> None:
    """Write a single vector-store record for the entire budget plan."""
    if not allocs:
        return

    try:
        # Derive month key from first allocation's period_start
        first_alloc = allocs[0]
        period_start: date = getattr(first_alloc, "period_start", None)
        if period_start:
            month_key = period_start.strftime("%Y-%m")
        else:
            from datetime import date as _date
            month_key = _date.today().strftime("%Y-%m")

        text = _budget_to_text(allocs, income)
        description = (
            f"Budget {month_key}: ${sum(a.allocated_amount for a in allocs):,.2f}"
            f" allocated, {len(allocs)} categories"
        )
        record_id = f"{agent_name}/{month_key}"
        embedding = embed(text)
        upsert(record_id, "budget", text, description, embedding)
    except Exception as exc:
        logger.warning("[memory.writer] Failed to write budget for %s: %s", agent_name, exc)


def _budget_to_text(allocs: list, income: float | None) -> str:
    """Serialise budget allocations + income to natural language."""
    try:
        total_allocated = sum(a.allocated_amount for a in allocs)
        n = len(allocs)

        # Derive month key from first allocation
        first_alloc = allocs[0]
        period_start: date = getattr(first_alloc, "period_start", None)
        month_key = period_start.strftime("%Y-%m") if period_start else "unknown"

        income_val = income if income is not None else 0.0

        lines = [
            f"Budget plan {month_key}: ${total_allocated:,.2f} allocated across {n} categories."
            f" Monthly income: ${income_val:,.2f}."
        ]

        for alloc in allocs:
            cat = alloc.category.value if hasattr(alloc.category, "value") else str(alloc.category)
            spent = alloc.spent_amount
            limit = alloc.allocated_amount
            if limit > 0:
                pct = abs((spent - limit) / limit * 100)
            else:
                pct = 0.0

            if spent > limit:
                lines.append(
                    f"  {cat} (${spent:,.2f} vs ${limit:,.2f} limit, +{pct:.0f}% over)."
                )
            else:
                remaining_pct = pct if limit > 0 else 0.0
                lines.append(
                    f"  {cat} (${spent:,.2f} vs ${limit:,.2f} limit, -{remaining_pct:.0f}% remaining)."
                )

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[memory.writer] Budget text serialisation failed, using fallback: %s", exc)
        # Fallback: JSON
        return json.dumps(
            {
                "budget_allocations": [
                    {
                        "category": str(getattr(a, "category", None)),
                        "allocated_amount": getattr(a, "allocated_amount", None),
                        "spent_amount": getattr(a, "spent_amount", None),
                    }
                    for a in allocs
                ],
                "monthly_income": income,
            },
            default=str,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_slug(name: str) -> str:
    """Convert e.g. 'Emergency Fund' → 'emergency-fund'."""
    slug = "".join(c if c.isalnum() or c == "-" else "-" for c in name.lower())
    # Collapse multiple dashes and strip leading/trailing dashes
    import re
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "goal"
