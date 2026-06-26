"""
Agent state scoping — defines what each agent can read and write.

Every agent node is wrapped with an AgentStateView that enforces its
declared scope at runtime: reading an undeclared field raises KeyError;
writing an undeclared field raises ValueError.  This prevents one agent
from accidentally polluting another agent's data.

AGENT_SCOPES is the single source of truth.  Adding a new field or agent
requires updating this dict and nothing else.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent field scopes
# ---------------------------------------------------------------------------
# "reads":  fields the agent is allowed to read from AppState.
# "writes": fields the agent is allowed to update in AppState.

AGENT_SCOPES: dict[str, dict[str, set[str]]] = {
    "expense_analysis": {
        "reads": {
            "messages",
            "transactions",
            "categorised_transactions",
            "current_date",
            "memory_context",
        },
        "writes": {
            "categorised_transactions",
            "spending_trends",
            "expense_analysis",
            "pending_confirmation",
            "messages",
        },
        "memory": set(),  # 分类结果不进入长期记忆
    },
    "budget_planning": {
        "reads": {
            "messages",
            "monthly_income",
            "current_date",
            "transactions",
            "categorised_transactions",
            "expense_analysis",
            "budget_allocations",
            "memory_context",
        },
        "writes": {
            "budget_allocations",
            "budget_progress",
            "budget_warnings",
            "budget_summary",
            "budget_request",
            "output_validation_result",
            "pending_confirmation",
            "messages",
            "security_events",
        },
        "memory": {"budget_allocations", "monthly_income"},
    },
    "goal_planning": {
        "reads": {
            "messages",
            "monthly_income",
            "current_date",
            "goals",
            "budget_allocations",
            "memory_context",
        },
        "writes": {
            "goals",
            "pending_confirmation",
            "hitl_rollback",
            "messages",
        },
        "memory": {"goals"},
    },
    "anomaly_detection": {
        "reads": {
            "messages",
            "categorised_transactions",
            "transactions",
            "memory_context",
        },
        "writes": {
            "anomaly_flags",
            "anomaly_explanation",
            "messages",
        },
        "memory": set(),  # 异常检测结果不持久化
    },
    "health_assessment": {
        "reads": {
            "messages",
            "categorised_transactions",
            "transactions",
            "monthly_income",
            "spending_trends",
            "alerts",
            "current_date",
            "memory_context",
        },
        "writes": {
            "health_summary",
            "alerts",
            "messages",
        },
        "memory": set(),  # 健康评估不持久化
    },
}


# ---------------------------------------------------------------------------
# Validation: ensure scope field names match actual AppState fields
# ---------------------------------------------------------------------------


def _validate_scopes() -> None:
    """
    Verify every field name in ``AGENT_SCOPES`` exists in ``AppState``.

    Raises ``ValueError`` on mismatch so mistakes (typos, leftover fields
    after rename) are caught at import time, not at runtime.
    """
    from app.state import AppState

    # TypedDict stores annotations in __annotations__
    # For Annotated types like messages, we get the full Annotated wrapper
    state_fields: set[str] = set(AppState.__annotations__.keys())

    for agent_name, scope in AGENT_SCOPES.items():
        for direction in ("reads", "writes", "memory"):
            if direction not in scope:
                continue
            for field in scope[direction]:
                if field not in state_fields:
                    raise ValueError(
                        f"AGENT_SCOPES['{agent_name}']['{direction}'] "
                        f"references '{field}', but no such field exists "
                        f"in AppState. Available fields: {sorted(state_fields)}"
                    )

    logger.debug(
        "state_view: validated %d agents against %d AppState fields",
        len(AGENT_SCOPES),
        len(state_fields),
    )


_validate_scopes()


# ---------------------------------------------------------------------------
# Scoped view
# ---------------------------------------------------------------------------


class AgentStateView:
    """
    Scoped read/write projection of AppState for a single agent node.

    Usage inside an agent::

        def run(view: AgentStateView, config=None) -> dict:
            txn = view.get("transactions")
            view.set("categorised_transactions", result)
            return view.updates
    """

    def __init__(self, state: dict, agent_name: str) -> None:
        scope = AGENT_SCOPES.get(agent_name)
        if scope is None:
            raise ValueError(
                f"Unknown agent '{agent_name}'. "
                f"Available: {list(AGENT_SCOPES)}"
            )
        self._readable: set[str] = scope["reads"]
        self._writable: set[str] = scope["writes"]
        self._memory_fields: set[str] = scope.get("memory", set())
        self._source: dict = state
        self._updates: dict = {}
        self._dirty_memory_fields: set[str] = set()
        self._agent_name: str = agent_name

    # -- read --

    def get(self, key: str, default=None):
        """Read a field from AppState.  Raises KeyError if not in reads scope."""
        if key not in self._readable:
            raise KeyError(
                f"Agent '{self._agent_name}' tried to read '{key}', "
                f"which is not in its reads scope. "
                f"Allowed reads: {sorted(self._readable)}"
            )
        return self._source.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._readable and key in self._source

    # -- write --

    def set(self, key: str, value) -> None:
        """Queue a field update.  Raises ValueError if not in writes scope."""
        if key not in self._writable:
            raise ValueError(
                f"Agent '{self._agent_name}' tried to write '{key}', "
                f"which is not in its writes scope. "
                f"Allowed writes: {sorted(self._writable)}"
            )
        self._updates[key] = value
        if key in self._memory_fields:
            self._dirty_memory_fields.add(key)

    # -- output --

    @property
    def updates(self) -> dict:
        """The collected state updates (read-only)."""
        return dict(self._updates)

    @property
    def dirty_memory_fields(self) -> frozenset[str]:
        """Fields in the agent's memory scope that were written this turn."""
        return frozenset(self._dirty_memory_fields)

    @property
    def agent_name(self) -> str:
        return self._agent_name
