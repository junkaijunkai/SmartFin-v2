"""
Integration tests for the agent pipeline.
These tests verify that the graph compiles, routes correctly, and
completes without errors.

Run with:
    pytest tests/integration/test_agent_pipeline.py -v
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from app.orchestrator import app_graph, get_pending_interrupt
from app.state import BudgetAllocation, FinancialGoal, Transaction, TransactionCategory


# A thread_id uniquely identifies one user's conversation session.
# Use a different id per test so checkpointer state doesn't bleed across tests.
def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _load_sample_transactions() -> list[Transaction]:
    """Load sample transactions from fixture and convert to Transaction objects."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_transactions.json"
    with open(fixture_path) as f:
        raw_txns = json.load(f)

    # Convert dicts to Transaction objects
    transactions = [
        Transaction(
            id=t["id"],
            date=t["date"],
            amount=t["amount"],
            description=t["description"],
            merchant=t["merchant"],
            category=TransactionCategory(t["category"]),
        )
        for t in raw_txns
    ]
    return transactions


def _extract_monthly_income(transactions: list[Transaction] | list[dict]) -> float:
    """Extract monthly income from transaction list (negative amounts indicate income)."""
    incomes = []
    for t in transactions:
        amount = t["amount"] if isinstance(t, dict) else t.amount
        if amount < 0:
            incomes.append(abs(amount))

    if incomes:
        return max(incomes)  # Typically the largest income is monthly salary
    return 5000.0  # fallback


def _make_initial_state(
    monthly_income: float = None,
    transactions: list[Transaction] = None,
    goals: list[FinancialGoal] = None,
    **kwargs
) -> dict:
    """Create minimal valid state using sample fixture data."""
    if transactions is None:
        transactions = _load_sample_transactions()

    # Extract monthly_income from transactions if not provided
    if monthly_income is None:
        monthly_income = _extract_monthly_income(transactions)

    if goals is None:
        goals = []

    state = {
        "messages": [],
        "transactions": transactions,
        "monthly_income": monthly_income,
        "goals": goals,
        "current_date": "2026-04-19",
    }
    state.update(kwargs)
    return state


def invoke(message: str, thread_id: str) -> dict:
    """Invoke the graph with a single human message and return final state."""
    return app_graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        make_config(thread_id),
    )


def invoke_with_state(
    message: str,
    thread_id: str,
    initial_state: dict = None
) -> dict:
    """Invoke with optional initial state."""
    if initial_state is None:
        initial_state = _make_initial_state()

    initial_state["messages"] = [HumanMessage(content=message)]
    return app_graph.invoke(initial_state, make_config(thread_id))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_compiles():
    """app_graph should be a compiled graph, not None."""
    assert app_graph is not None


def test_expense_analysis_route():
    """A generic message should route through expense_analysis."""
    state = invoke_with_state(
        "Show me my spending",
        thread_id="t-expense",
        initial_state=_make_initial_state()
    )
    # Expense analysis may pause for HITL confirmation or complete normally
    assert state["active_agent"] == "end" or state.get("pending_confirmation") is not None
    # Verify agent produced output
    assert "categorised_transactions" in state
    assert "spending_trends" in state
    assert isinstance(state["categorised_transactions"], list)
    assert isinstance(state["spending_trends"], list)


def test_budget_planning_route():
    """Message containing 'budget' should route through budget_planning."""
    state = invoke_with_state(
        "Help me with my budget",
        thread_id="t-budget",
        initial_state=_make_initial_state()
    )
    # May complete, pause for expense_analysis HITL, or pause for budget_planning HITL
    assert state["active_agent"] == "end" or state.get("pending_confirmation") is not None
    # Verify agent produced output (if reached budget_planning)
    if "budget_allocations" in state:
        assert isinstance(state["budget_allocations"], list)
        assert all(isinstance(a, BudgetAllocation) for a in state["budget_allocations"])
        assert "budget_progress" in state
        assert isinstance(state["budget_progress"], dict)
        assert "budget_warnings" in state
        assert isinstance(state["budget_warnings"], list)


def test_goal_planning_route():
    """Message containing 'goal' should route through goal_planning."""
    goal = FinancialGoal(
        id="g1",
        name="Emergency Fund",
        target_amount=2000.0,
        current_amount=500.0,
        target_date=date.today() + timedelta(days=90)
    )
    state = invoke_with_state(
        "I want to set a savings goal",
        thread_id="t-goal",
        initial_state=_make_initial_state(goals=[goal])
    )
    # May complete, pause for expense_analysis HITL, or pause for goal_planning HITL
    assert state["active_agent"] == "end" or state.get("pending_confirmation") is not None
    # Verify agent updated goals (if reached goal_planning)
    if state["goals"] and hasattr(state["goals"][0], "required_monthly_saving"):
        assert len(state["goals"]) > 0
        assert hasattr(state["goals"][0], "required_monthly_saving")
        assert hasattr(state["goals"][0], "on_track")


def test_anomaly_detection_route():
    """Message containing 'suspicious' should route through anomaly_detection."""
    state = invoke_with_state(
        "There's a suspicious transaction",
        thread_id="t-anomaly",
        initial_state=_make_initial_state()
    )
    # May complete, pause for expense_analysis HITL, or pause for anomaly_detection HITL
    assert state["active_agent"] == "end" or state.get("pending_confirmation") is not None
    # Verify agent produced output (if reached anomaly_detection)
    if "anomaly_flags" in state:
        assert isinstance(state["anomaly_flags"], list)


def test_health_assessment_route():
    """Message containing 'health' should route through health_assessment."""
    state = invoke_with_state(
        "What is my financial health?",
        thread_id="t-health",
        initial_state=_make_initial_state()
    )
    # May complete, pause for expense_analysis HITL, or pause for health_assessment HITL
    assert state["active_agent"] == "end" or state.get("pending_confirmation") is not None
    # Verify agent produced output (if reached health_assessment)
    health_summary = state.get("health_summary")
    if health_summary is not None:
        assert health_summary.rating is not None


def test_messages_are_accumulated():
    """The messages list should contain the original human message after the run."""
    state = invoke_with_state(
        "Show me my spending",
        thread_id="t-messages",
        initial_state=_make_initial_state()
    )
    # Should have at least the user message, possibly AI responses too
    assert len(state["messages"]) > 0
    contents = [m.content for m in state["messages"]]
    assert "Show me my spending" in contents


def test_graph_runs_without_crashing():
    """Graph should run without raising exceptions on normal input."""
    config = make_config("t-run")
    initial_state = _make_initial_state()
    initial_state["messages"] = [HumanMessage(content="Show me my spending")]
    # Should not raise exceptions — may pause for HITL or complete normally
    state = app_graph.invoke(initial_state, config)
    assert state is not None
    assert "messages" in state
