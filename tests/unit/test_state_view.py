"""Unit tests for AgentStateView — field isolation and scope enforcement."""
from __future__ import annotations

import pytest
from app.orchestrator.state_view import AgentStateView, AGENT_SCOPES


class TestAgentStateViewBasics:
    """Basic read/write within allowed scopes."""

    def test_read_allowed_field(self):
        state = {"messages": ["hello"], "transactions": []}
        view = AgentStateView(state, "anomaly_detection")
        assert view.get("messages") == ["hello"]

    def test_read_with_default(self):
        state = {}
        view = AgentStateView(state, "expense_analysis")
        assert view.get("transactions") is None
        assert view.get("transactions", default=[]) == []

    def test_set_allowed_field(self):
        state = {}
        view = AgentStateView(state, "goal_planning")
        view.set("goals", [])
        assert view.updates == {"goals": []}

    def test_set_then_get_unchanged_source(self):
        """set() queues the update but does NOT modify the source dict."""
        state = {"messages": []}
        view = AgentStateView(state, "expense_analysis")
        view.set("messages", ["new"])
        assert state["messages"] == []  # source unchanged
        assert view.updates["messages"] == ["new"]

    def test_updates_read_only_copy(self):
        """updates returns a copy; mutating it should not affect internal state."""
        state = {}
        view = AgentStateView(state, "anomaly_detection")
        view.set("anomaly_flags", [1, 2, 3])
        updates = view.updates
        updates["anomaly_flags"] = []
        assert view.updates["anomaly_flags"] == [1, 2, 3]

    def test_multiple_sets_accumulate(self):
        state = {"messages": []}
        view = AgentStateView(state, "goal_planning")
        view.set("goals", ["g1"])
        view.set("pending_confirmation", {"action": "approve"})
        assert view.updates == {
            "goals": ["g1"],
            "pending_confirmation": {"action": "approve"},
        }

    def test_agent_name(self):
        view = AgentStateView({}, "expense_analysis")
        assert view.agent_name == "expense_analysis"

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            AgentStateView({}, "nonexistent_agent")

    def test_contains_checks_read_scope_and_source(self):
        state = {"messages": []}
        view = AgentStateView(state, "expense_analysis")
        assert "messages" in view        # in reads + in source
        assert "transactions" not in view  # in reads but not in source
        assert "goals" not in view         # not in reads


class TestAgentStateViewReadIsolation:
    """Reading undeclared fields should raise."""

    def test_read_outside_scope_raises(self):
        state = {"goals": []}
        view = AgentStateView(state, "expense_analysis")
        with pytest.raises(KeyError, match="tried to read.*goals"):
            view.get("goals")

    @pytest.mark.parametrize("agent", list(AGENT_SCOPES.keys()))
    def test_each_agent_can_read_its_scoped_fields(self, agent):
        """All permitted reads should succeed without error."""
        scope = AGENT_SCOPES[agent]
        state = {field: None for field in scope["reads"]}
        view = AgentStateView(state, agent)
        for field in scope["reads"]:
            # Should not raise
            view.get(field)

    @pytest.mark.parametrize("agent", list(AGENT_SCOPES.keys()))
    def test_each_agent_cannot_read_outside_scope(self, agent):
        """Reading any field not in reads scope should raise KeyError."""
        scope = AGENT_SCOPES[agent]
        # Pick a field that no agent reads — "budget_allocations" is only
        # in budget_planning reads, so test from expense_analysis.
        all_fields = {"messages", "transactions", "monthly_income", "current_date",
                      "goals", "categorised_transactions", "spending_trends",
                      "expense_analysis", "budget_allocations"}
        safe_fields = scope["reads"]
        forbidden = all_fields - safe_fields
        if not forbidden:
            return  # agent reads everything
        state = {}
        view = AgentStateView(state, agent)
        with pytest.raises(KeyError):
            view.get(next(iter(forbidden)))


class TestAgentStateViewWriteIsolation:
    """Writing undeclared fields should raise."""

    def test_write_outside_scope_raises(self):
        state = {}
        view = AgentStateView(state, "anomaly_detection")
        with pytest.raises(ValueError, match="tried to write.*goals"):
            view.set("goals", [])

    @pytest.mark.parametrize("agent", list(AGENT_SCOPES.keys()))
    def test_each_agent_can_write_its_scoped_fields(self, agent):
        """All permitted writes should succeed without error."""
        scope = AGENT_SCOPES[agent]
        state = {}
        view = AgentStateView(state, agent)
        for field in scope["writes"]:
            view.set(field, None)  # Should not raise

    @pytest.mark.parametrize("agent", list(AGENT_SCOPES.keys()))
    def test_each_agent_cannot_write_outside_scope(self, agent):
        """Writing any field not in writes scope should raise ValueError."""
        scope = AGENT_SCOPES[agent]
        all_writable = set()
        for s in AGENT_SCOPES.values():
            all_writable |= s["writes"]
        forbidden = all_writable - scope["writes"]
        if not forbidden:
            return
        state = {}
        view = AgentStateView(state, agent)
        with pytest.raises(ValueError):
            view.set(next(iter(forbidden)), None)
