"""
File-system state store — durable backup of AppState after agent execution.

Backup layer for the checkpoint system.  Each thread gets its own directory
under ``.smartfin_memory/<thread_id>/``.  On every turn ``memory_saver``
writes agent-generated state to JSON files; ``confirm_node`` also writes
after HITL approval.

The new long-term memory system (`.smartfin/memory/` with `.md` files) is
separate and lives in ``app/memory/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.state import (
    Alert,
    AnomalyFlag,
    BudgetAllocation,
    FinancialGoal,
    HealthSummary,
    SpendingTrend,
    Transaction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEMORY_ROOT = _REPO_ROOT / ".smartfin_memory"

_FILE_MAP: dict[str, tuple[str, str]] = {
    "goals": ("goals.json", "goals"),
    "budget_allocations": ("budgets.json", "budget_allocations"),
    "budget_progress": ("budgets.json", "budget_progress"),
    "budget_warnings": ("budgets.json", "budget_warnings"),
    "budget_summary": ("budgets.json", "budget_summary"),
    "budget_request": ("budgets.json", "budget_request"),
    "categorised_transactions": ("expense_profile.json", "categorised_transactions"),
    "spending_trends": ("expense_profile.json", "spending_trends"),
    "expense_analysis": ("expense_profile.json", "expense_analysis"),
    "anomaly_flags": ("anomalies.json", "anomaly_flags"),
    "anomaly_explanation": ("anomalies.json", "anomaly_explanation"),
    "health_summary": ("health.json", "health_summary"),
    "alerts": ("health.json", "alerts"),
    "transactions": ("transactions.json", "transactions"),
    "monthly_income": ("transactions.json", "monthly_income"),
}

_MODEL_FIELDS: dict[str, type] = {
    "goals": FinancialGoal,
    "budget_allocations": BudgetAllocation,
    "categorised_transactions": Transaction,
    "transactions": Transaction,
    "spending_trends": SpendingTrend,
    "anomaly_flags": AnomalyFlag,
    "health_summary": HealthSummary,
    "alerts": Alert,
}

_EXCLUDED_FIELDS = {
    "messages", "current_date", "active_agent", "pending_intent",
    "pending_confirmation", "input_filter_result", "output_validation_result",
    "security_events", "hitl_rollback", "hitl_decision", "last_intent",
}


def _sanitise(thread_id: str) -> str:
    return "".join(c for c in thread_id if c.isalnum() or c in "-_")


class MemoryStore:
    """Per-thread file-system state persistence (backup only)."""

    def __init__(self, thread_id: str) -> None:
        self._thread_dir = _MEMORY_ROOT / _sanitise(thread_id)

    def save_state(self, state: dict) -> None:
        """Persist known fields from ``state`` to JSON files."""
        self._thread_dir.mkdir(parents=True, exist_ok=True)
        file_buckets: dict[str, dict] = {}
        for field, fname in _FILE_MAP.items():
            if field in _EXCLUDED_FIELDS:
                continue
            filename, key = fname
            if field not in state:
                continue
            value = state[field]
            if value is None or (isinstance(value, list) and not value):
                continue
            file_buckets.setdefault(filename, {})[key] = self._serialise(field, value)
        for filename, payload in file_buckets.items():
            path = self._thread_dir / filename
            try:
                path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.error("[memory_store] failed to write %s: %s", filename, exc)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _serialise(self, field: str, value):
        model = _MODEL_FIELDS.get(field)
        if model is None:
            return value
        if isinstance(value, list):
            return [v.model_dump(mode="json") for v in value]
        if isinstance(value, model):
            return value.model_dump(mode="json")
        return value

