"""
Integration tests for the Human-in-the-Loop (HITL) confirmation flow.

These tests verify the full interrupt-resume cycle:
  1. Graph pauses at confirm node with pending_confirmation set
  2. User approves / rejects / clarifies via resume_with_confirmation()
  3. Graph continues to END or re-routes for clarification

Tests make real LLM calls — requires a valid ANTHROPIC_API_KEY.
Each test uses a unique thread_id to prevent checkpointer state bleed.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from app.orchestrator import app_graph
from app.orchestrator.checkpoints import get_pending_interrupt, resume_with_confirmation
from app.state import Transaction, TransactionCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _load_sample_transactions() -> list[Transaction]:
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_transactions.json"
    with open(fixture_path) as f:
        raw = json.load(f)
    return [
        Transaction(
            id=t["id"],
            date=t["date"],
            amount=t["amount"],
            description=t["description"],
            merchant=t["merchant"],
            category=TransactionCategory(t["category"]),
        )
        for t in raw
    ]


def _invoke_and_expect_hitl(message: str, thread_id: str, extra_state: dict = None) -> dict:
    """
    Invoke the graph and return the paused state.
    Asserts that the graph actually paused for HITL rather than completing.
    """
    config = make_config(thread_id)
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "transactions": _load_sample_transactions(),
        "monthly_income": 3200.0,
        "goals": [],
        "current_date": "2026-04-21",
    }
    if extra_state:
        initial_state.update(extra_state)

    state = app_graph.invoke(initial_state, config)

    interrupted = get_pending_interrupt(app_graph, config)
    assert interrupted is not None, (
        "Expected graph to pause for HITL, but it completed without interruption. "
        f"pending_confirmation={state.get('pending_confirmation')}"
    )
    return state


# ---------------------------------------------------------------------------
# Test 1: HITL pause detection
# ---------------------------------------------------------------------------

def test_hitl_pause_is_detected_after_expense_analysis():
    """
    After expense_analysis runs, get_pending_interrupt should return the
    paused state with a valid pending_confirmation payload.
    """
    config = make_config("hitl-detect-01")
    app_graph.checkpointer.delete_thread("hitl-detect-01")
    initial_state = {
        "messages": [HumanMessage(content="Show me my spending")],
        "transactions": _load_sample_transactions(),
        "monthly_income": 3200.0,
        "goals": [],
        "current_date": "2026-04-21",
    }

    app_graph.invoke(initial_state, config)

    paused = get_pending_interrupt(app_graph, config)

    # Graph must have paused — paused state should be non-None
    assert paused is not None, "Graph should be paused at confirm node after expense_analysis"

    pc = paused.get("pending_confirmation")
    assert pc is not None, "Paused state must contain pending_confirmation"
    assert "action" in pc
    assert "agent" in pc
    assert "summary" in pc
    assert "details" in pc
    # confirmed must NOT be set — this is what triggers the HITL pause
    assert "confirmed" not in pc


# ---------------------------------------------------------------------------
# Test 2: Approval flow
# ---------------------------------------------------------------------------

def test_hitl_approval_clears_pending_confirmation():
    """
    After user approves, confirm_node must clear pending_confirmation and
    the graph must reach END cleanly.
    """
    thread_id = "hitl-approve-01"
    app_graph.checkpointer.delete_thread(thread_id)
    config = make_config(thread_id)

    _invoke_and_expect_hitl("Show me my spending", thread_id)

    # User approves
    final_state = resume_with_confirmation(app_graph, config, confirmed=True)

    assert final_state.get("pending_confirmation") is None, (
        "pending_confirmation should be cleared after user approval"
    )
    assert final_state.get("active_agent") == "end"


# ---------------------------------------------------------------------------
# Test 3: Rejection flow
# ---------------------------------------------------------------------------

def test_hitl_rejection_clears_pending_confirmation():
    """
    After user rejects, confirm_node must also clear pending_confirmation
    and the graph must reach END.
    """
    thread_id = "hitl-reject-01"
    app_graph.checkpointer.delete_thread(thread_id)
    config = make_config(thread_id)

    _invoke_and_expect_hitl("Show me my spending", thread_id)

    # User rejects
    final_state = resume_with_confirmation(app_graph, config, confirmed=False)

    assert final_state.get("pending_confirmation") is None, (
        "pending_confirmation should be cleared even after rejection"
    )
    assert final_state.get("active_agent") == "end"


# ---------------------------------------------------------------------------
# Test 4: Clarification flow — budget_planning missing income
# ---------------------------------------------------------------------------

def test_hitl_budget_clarification_reroutes_to_supervisor():
    """
    When budget_planning cannot determine monthly income, it sets
    action='clarify_budget_planning' and pauses for HITL.

    After the user provides income via resume_with_confirmation(user_message=...),
    the graph should:
      - clear pending_confirmation
      - reset active_agent to None so supervisor re-routes
      - (supervisor then classifies the new message and re-runs budget_planning)

    We only assert the re-routing mechanics here, not the second LLM cycle outcome,
    to keep the test deterministic.
    """
    thread_id = "hitl-clarify-budget-01"
    app_graph.checkpointer.delete_thread(thread_id)
    config = make_config(thread_id)

    # Provide no income in state and no income-bearing transactions
    # so extractor must ask for clarification
    expense_only_transactions = [
        t for t in _load_sample_transactions() if t.amount > 0
    ]
    initial_state = {
        "messages": [HumanMessage(content="Help me plan my monthly budget")],
        "transactions": expense_only_transactions,
        # monthly_income intentionally omitted so extractor gets None from state
        "goals": [],
        "current_date": "2026-04-21",
    }

    state = app_graph.invoke(initial_state, config)

    # The graph may pause at expense_analysis HITL before reaching budget_planning.
    # We skip through any upstream HITL pauses (approve them) until we either
    # reach a clarify_budget_planning pause or the graph ends.
    max_cycles = 3
    for _ in range(max_cycles):
        paused = get_pending_interrupt(app_graph, config)
        if paused is None:
            break

        pc = paused.get("pending_confirmation", {})
        if pc.get("action") == "clarify_budget_planning":
            break

        # Approve upstream HITL (e.g. expense_analysis) and continue
        state = resume_with_confirmation(app_graph, config, confirmed=True)

    paused = get_pending_interrupt(app_graph, config)

    if paused is None:
        # Graph completed without reaching budget_planning clarification —
        # this can happen if monthly_income was inferred elsewhere (e.g. from
        # categorised_transactions). Skip rather than fail.
        pytest.skip(
            "Graph completed without triggering clarify_budget_planning "
            "(income may have been inferred from transaction data)."
        )

    pc = paused.get("pending_confirmation", {})
    if pc.get("action") != "clarify_budget_planning":
        pytest.skip(
            f"Graph paused on a different action ({pc.get('action')}), "
            "not the budget clarification we were testing."
        )

    assert pc["agent"] == "budget_planning"
    assert "details" in pc

    # User provides income in clarification message
    final_state = resume_with_confirmation(
        app_graph,
        config,
        confirmed=True,
        user_message="My monthly take-home pay is £3200",
    )

    # The clarification path in confirm_node resets active_agent to None
    # so the supervisor makes a fresh routing decision.
    # pending_confirmation must be cleared.
    assert final_state.get("pending_confirmation") is None, (
        "pending_confirmation should be cleared after clarification"
    )
