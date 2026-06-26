"""Memory writer — persists agent task outputs as natural-language embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from app.memory.embeddings import embed_batch
from app.memory.vector_store import batch_upsert, fetch_hashes_batch

logger = logging.getLogger(__name__)

# (record_id, content_text, description, expires_at)
_Record = tuple[str, str, str, datetime | None]
# _Record + SHA-256 content hash (produced by _filter_changed_batch)
_RecordH = tuple[str, str, str, datetime | None, str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_task_memory(agent_name: str, state: dict, dirty_fields: set[str] | None = None) -> None:
    """Write one memory record per completed task to the vector store.

    Only embeds records whose content has changed since the last write
    (detected via SHA-256 hash comparison in a single batch DB query).

    ``dirty_fields`` — when provided, limits processing to memory-scoped fields
    that the agent actually wrote this turn.  Pass ``None`` (HITL / legacy path)
    to process all memory-scoped fields unconditionally.

    Failures are caught and logged as warnings — this function is non-critical
    and must never raise.
    """
    try:
        from app.orchestrator.state_view import AGENT_SCOPES

        scope = AGENT_SCOPES.get(agent_name, {})
        memory_fields: set[str] = scope.get("memory", set())

        if not memory_fields:
            return

        # Narrow to fields that were actually dirtied this turn (when known).
        fields_to_process: set[str] = (
            memory_fields & dirty_fields if dirty_fields is not None else memory_fields
        )
        if not fields_to_process:
            return

        if "goals" in fields_to_process:
            goals = state.get("goals") or []
            _write_goals(goals, agent_name)

        # Budget memory is triggered by either allocation or income change.
        budget_triggers = {"budget_allocations", "monthly_income"}
        if budget_triggers & fields_to_process:
            allocs = state.get("budget_allocations") or []
            income = state.get("monthly_income")
            _write_budget(allocs, income, agent_name)

    except Exception as exc:
        logger.warning("[memory.writer] write_task_memory failed for %s: %s", agent_name, exc)


# ---------------------------------------------------------------------------
# Private helpers — change detection and batch I/O
# ---------------------------------------------------------------------------


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _filter_changed_batch(candidates: list[_Record]) -> list[_RecordH]:
    """Return only changed records, with their SHA-256 hash — single batch DB query."""
    if not candidates:
        return []
    ids = [r[0] for r in candidates]
    try:
        stored_hashes = fetch_hashes_batch(ids)
    except Exception as exc:
        logger.warning("[memory.writer] fetch_hashes_batch failed: %s — treating all as changed", exc)
        stored_hashes = {}
    changed: list[_RecordH] = []
    for record_id, text, description, expires_at in candidates:
        new_hash = _content_hash(text)
        if stored_hashes.get(record_id) != new_hash:
            changed.append((record_id, text, description, expires_at, new_hash))
    return changed


def _batch_write(records: list[_RecordH], memory_type: str) -> None:
    """Embed all records in one API call and upsert in one DB transaction."""
    texts = [r[1] for r in records]
    embeddings = embed_batch(texts)
    db_records = [
        {
            "id": record_id,
            "memory_type": memory_type,
            "content": text,
            "description": description,
            "embedding": embedding,
            "expires_at": expires_at,
            "content_hash": content_hash,
        }
        for (record_id, text, description, expires_at, content_hash), embedding in zip(records, embeddings)
    ]
    batch_upsert(db_records)
    logger.debug("[memory.writer] Wrote %d %s record(s)", len(db_records), memory_type)


# ---------------------------------------------------------------------------
# Private helpers — expiration
# ---------------------------------------------------------------------------


def _goal_expires_at(goal: Any) -> datetime | None:
    """Expire 30 days after goal.target_date (in UTC). Returns None on failure."""
    try:
        target_date: date = goal.target_date
        return datetime.combine(target_date, time.max, tzinfo=timezone.utc) + timedelta(days=30)
    except Exception:
        return None


def _budget_expires_at(period_start: date | None) -> datetime | None:
    """Expire 90 days after period_start (in UTC). Returns None on failure."""
    try:
        if period_start is None:
            return None
        return datetime.combine(period_start, time.min, tzinfo=timezone.utc) + timedelta(days=90)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Private helpers — goal persistence
# ---------------------------------------------------------------------------


def _write_goals(goals: list, agent_name: str) -> None:
    """Build candidates, filter unchanged, batch-embed and upsert."""
    if not goals:
        return

    candidates: list[_Record] = []
    for goal in goals:
        try:
            text = _goal_to_text(goal)
            description = (
                f"Goal '{goal.name}': ${goal.current_amount:,.2f}/${goal.target_amount:,.2f}"
                f" by {goal.target_date},"
                f" {'on track' if goal.on_track else 'behind'}"
            )
            record_id = f"{agent_name}/{_safe_slug(goal.name)}"
            candidates.append((record_id, text, description, _goal_expires_at(goal)))
        except Exception as exc:
            logger.warning("[memory.writer] Failed to build goal candidate '%s': %s", getattr(goal, "name", "?"), exc)

    changed = _filter_changed_batch(candidates)
    if not changed:
        logger.debug("[memory.writer] All goal records unchanged, skipping embed")
        return

    _batch_write(changed, "goal")


def _goal_to_text(goal: Any) -> str:
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
    """Build candidate, filter if unchanged, batch-embed and upsert."""
    if not allocs:
        return

    try:
        first_alloc = allocs[0]
        period_start: date | None = getattr(first_alloc, "period_start", None)
        if period_start:
            month_key = period_start.strftime("%Y-%m")
        else:
            from datetime import date as _date
            month_key = _date.today().strftime("%Y-%m")
            period_start = None

        text = _budget_to_text(allocs, income)
        description = (
            f"Budget {month_key}: ${sum(a.allocated_amount for a in allocs):,.2f}"
            f" allocated, {len(allocs)} categories"
        )
        record_id = f"{agent_name}/{month_key}"
        candidate: _Record = (record_id, text, description, _budget_expires_at(period_start))

        changed = _filter_changed_batch([candidate])
        if not changed:
            logger.debug("[memory.writer] Budget record unchanged, skipping embed")
            return

        _batch_write(changed, "budget")
    except Exception as exc:
        logger.warning("[memory.writer] Failed to write budget for %s: %s", agent_name, exc)


def _budget_to_text(allocs: list, income: float | None) -> str:
    """Serialise budget allocations + income to natural language."""
    try:
        total_allocated = sum(a.allocated_amount for a in allocs)
        n = len(allocs)

        first_alloc = allocs[0]
        period_start: date | None = getattr(first_alloc, "period_start", None)
        month_key = period_start.strftime("%Y-%m") if period_start else "unknown"

        income_val = income if income is not None else 0.0

        lines = [
            f"Budget plan {month_key}: ${total_allocated:,.2f} allocated across {n} categories."
            f" Monthly income: ${income_val:,.2f}."
        ]

        # Sort by category name to guarantee deterministic hash regardless of
        # iteration order over the set in evaluate_budget_progress().
        sorted_allocs = sorted(allocs, key=lambda a: a.category.value if hasattr(a.category, "value") else str(a.category))
        for alloc in sorted_allocs:
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
    import re
    slug = "".join(c if c.isalnum() or c == "-" else "-" for c in name.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "goal"
