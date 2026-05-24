"""Unit tests for the Goal Planning agent — tracker, extractor, and agent node."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from app.state import AppState, FinancialGoal, BudgetAllocation, TransactionCategory
from app.agents.goal_planning.tracker import (
    calculate_months_remaining,
    calculate_required_monthly_saving,
)
import app.agents.goal_planning.agent as goal_agent_module
from app.agents.goal_planning.agent import run as goal_planning_run
from app.orchestrator.state_view import AgentStateView
import app.agents.goal_planning.extractor as extractor_module
from app.agents.goal_planning.extractor import (
    GoalExtractionResult,
    extract_goal_from_message,
)


# ---------------------------------------------------------------------------
# tracker.py tests
# ---------------------------------------------------------------------------


def test_calculate_months_remaining_future_date():
    target_date = date.today() + timedelta(days=90)
    assert calculate_months_remaining(target_date) == 3


def test_calculate_months_remaining_past_date():
    target_date = date.today() - timedelta(days=10)
    assert calculate_months_remaining(target_date) == 1


def test_calculate_required_monthly_saving_basic():
    goal = FinancialGoal(
        id="g1", name="Emergency Fund", target_amount=1200.0,
        current_amount=0.0, target_date=date.today() + timedelta(days=120),
    )
    assert calculate_required_monthly_saving(goal) == 300.0


def test_calculate_required_monthly_saving_already_completed():
    goal = FinancialGoal(
        id="g2", name="Completed", target_amount=1000.0,
        current_amount=1200.0, target_date=date.today() + timedelta(days=60),
    )
    assert calculate_required_monthly_saving(goal) == 0.0


# ---------------------------------------------------------------------------
# extractor.py tests
# ---------------------------------------------------------------------------


def test_fallback_extract_complete_laptop_goal():
    result = extractor_module._fallback_extract(
        "I want to save 8000 by 2027-06-01 for a laptop."
    )
    assert result.is_goal_intent is True
    assert result.name == "Laptop Fund"
    assert result.target_amount == 8000.0
    assert result.target_date == date(2027, 6, 1)
    assert result.missing_fields == []


def test_fallback_extract_missing_amount():
    result = extractor_module._fallback_extract(
        "I want to save for a laptop by 2027-06-01."
    )
    assert result.is_goal_intent is True
    assert result.target_amount is None
    assert "target_amount" in result.missing_fields


def test_fallback_extract_missing_date():
    result = extractor_module._fallback_extract(
        "I want to save 5000 for travel."
    )
    assert result.is_goal_intent is True
    assert result.target_amount == 5000.0
    assert result.target_date is None
    assert "target_date" in result.missing_fields


def test_fallback_extract_non_goal():
    result = extractor_module._fallback_extract("Hello, how are you?")
    assert result.is_goal_intent is False


def test_fallback_extract_emergency():
    result = extractor_module._fallback_extract(
        "I want to build an emergency fund of 10000 by 2027-12-31."
    )
    assert result.name == "Emergency Fund"


def test_extract_empty_input():
    result, ok = extract_goal_from_message("   ")
    assert ok is True and result.is_goal_intent is False


def test_extract_llm_success(monkeypatch):
    expected = GoalExtractionResult(
        is_goal_intent=True, name="Laptop Fund", target_amount=8000.0,
        target_date=date(2027, 6, 1), current_amount=1000.0, missing_fields=[],
    )

    class FakeStructured:
        def invoke(self, messages):
            return expected

    class FakeLLM:
        def __init__(self, **kw):
            pass
        def with_structured_output(self, schema):
            return FakeStructured()

    monkeypatch.setattr(extractor_module, "ChatAnthropic", FakeLLM)
    result, ok = extract_goal_from_message("I want to save 8000 by 2027-06-01 for a laptop.")
    assert ok is True
    assert result == expected


def test_extract_llm_failure_fallback(monkeypatch):
    class FakeLLM:
        def __init__(self, **kw):
            raise RuntimeError("fail")
        def with_structured_output(self, schema):
            return self

    monkeypatch.setattr(extractor_module, "ChatAnthropic", FakeLLM)
    result, ok = extract_goal_from_message(
        "I want to save 8000 by 2027-06-01 for a laptop."
    )
    assert ok is False
    assert result.is_goal_intent is True


# ---------------------------------------------------------------------------
# Helpers for agent tests
# ---------------------------------------------------------------------------


def _make_state(
    goals: list[FinancialGoal],
    monthly_income: float,
    budget_allocations: list[BudgetAllocation],
    messages: list | None = None,
) -> AppState:
    return {
        "messages": messages or [],
        "transactions": [],
        "monthly_income": monthly_income,
        "categorised_transactions": [],
        "spending_trends": [],
        "budget_allocations": budget_allocations,
        "goals": goals,
        "anomaly_flags": [],
        "health_summary": None,
        "alerts": [],
        "pending_confirmation": None,
        "active_agent": "goal_planning",
        "pending_intent": None,
    }


class DummyMessage:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return f"DummyMessage(content={self.content})"


def _run_with_state(state: dict) -> dict:
    """Wrap state in AgentStateView and call the agent."""
    return goal_planning_run(AgentStateView(state, "goal_planning"))


def test_get_latest_message_text_empty():
    state = _make_state(goals=[], monthly_income=0.0, budget_allocations=[], messages=[])
    assert goal_agent_module._get_latest_message_text(
        AgentStateView(state, "goal_planning")
    ) == ""


def test_get_latest_message_text_returns_content():
    state = _make_state(
        goals=[], monthly_income=0.0, budget_allocations=[],
        messages=[DummyMessage("I want to save 5000 for travel.")],
    )
    assert goal_agent_module._get_latest_message_text(
        AgentStateView(state, "goal_planning")
    ) == "I want to save 5000 for travel."


# ---------------------------------------------------------------------------
# Agent node tests  (mock run_react_loop, test pre/post processing)
# ---------------------------------------------------------------------------


def _mock_react(tc_updates: dict):
    """Return side-effect for @patch('...run_react_loop') that fills tool_ctx."""

    def side_effect(*args, **kwargs):
        tc = kwargs.get("tool_ctx")
        if tc is not None:
            tc.update(tc_updates)
        return MagicMock(), []

    return side_effect


@patch("app.agents.goal_planning.agent.run_react_loop")
def test_agent_updates_goal_fields(mock_react):
    mock_react.side_effect = _mock_react({
        "extraction": GoalExtractionResult(is_goal_intent=False, missing_fields=[]),
        "extraction_llm_ok": True,
    })

    goal = FinancialGoal(
        id="g1", name="Emergency Fund", target_amount=1200.0,
        current_amount=0.0, target_date=date.today() + timedelta(days=120),
    )
    budget = BudgetAllocation(
        category=TransactionCategory.FOOD, allocated_amount=600.0,
        spent_amount=500.0, period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )
    state = _make_state(goals=[goal], monthly_income=2000.0, budget_allocations=[budget])
    result = _run_with_state(state)

    assert len(result["goals"]) == 1
    assert result["goals"][0].required_monthly_saving == 300.0
    assert result["goals"][0].on_track is True


@patch("app.agents.goal_planning.agent.run_react_loop")
def test_agent_creates_new_goal_and_sets_hitl(mock_react):
    target = date.today() + timedelta(days=120)
    mock_react.side_effect = _mock_react({
        "extraction": GoalExtractionResult(
            is_goal_intent=True, name="Laptop Fund", target_amount=2400.0,
            target_date=target, current_amount=0.0, missing_fields=[],
        ),
        "new_goal": FinancialGoal(
            id="g-new", name="Laptop Fund", target_amount=2400.0,
            current_amount=0.0, target_date=target,
        ),
        "new_goal_added": True,
        "extraction_llm_ok": True,
    })

    budget = BudgetAllocation(
        category=TransactionCategory.SHOPPING, allocated_amount=800.0,
        spent_amount=700.0, period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )
    state = _make_state(
        goals=[], monthly_income=1500.0, budget_allocations=[budget],
        messages=[DummyMessage("I want to save 2400 for a laptop.")],
    )
    result = _run_with_state(state)

    assert result["pending_confirmation"]["action"] == "approve_goal_planning"
    assert "confirmed" not in result["pending_confirmation"]
    assert len(result["goals"]) == 1


@patch("app.agents.goal_planning.agent.run_react_loop")
def test_agent_marks_goal_behind_schedule(mock_react):
    mock_react.side_effect = _mock_react({
        "extraction": GoalExtractionResult(is_goal_intent=False, missing_fields=[]),
        "extraction_llm_ok": True,
    })

    goal = FinancialGoal(
        id="g3", name="Travel Fund", target_amount=5000.0,
        current_amount=0.0, target_date=date.today() + timedelta(days=60),
    )
    budget = BudgetAllocation(
        category=TransactionCategory.ENTERTAINMENT, allocated_amount=1000.0,
        spent_amount=900.0, period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )
    state = _make_state(
        goals=[goal], monthly_income=1000.0, budget_allocations=[budget],
    )
    result = _run_with_state(state)
    assert result["goals"][0].on_track is False


@patch("app.agents.goal_planning.agent.run_react_loop")
def test_agent_returns_clarification_when_fields_missing(mock_react):
    mock_react.side_effect = _mock_react({
        "extraction": GoalExtractionResult(
            is_goal_intent=True, name="Laptop Fund",
            target_amount=None, target_date=None, current_amount=None,
            missing_fields=["target_amount", "target_date"],
        ),
        "extraction_llm_ok": False,
    })

    state = _make_state(
        goals=[], monthly_income=2000.0, budget_allocations=[],
        messages=[DummyMessage("I want to save for a laptop.")],
    )
    result = _run_with_state(state)

    assert result.get("goals", []) == []
    pc = result["pending_confirmation"]
    assert pc["action"] == "clarify_goal_planning"
    assert pc["goal_extraction_confidence"] == "fallback"
    assert "Missing fields: target_amount, target_date" in pc["details"][1]


@patch("app.agents.goal_planning.agent.run_react_loop")
def test_agent_creates_goal_from_complete_extraction(mock_react):
    target = date.today() + timedelta(days=365)
    new_goal = FinancialGoal(
        id="g-new", name="Laptop Fund", target_amount=8000.0,
        current_amount=1000.0, target_date=target,
    )
    mock_react.side_effect = _mock_react({
        "extraction": GoalExtractionResult(
            is_goal_intent=True, name="Laptop Fund", target_amount=8000.0,
            target_date=target, current_amount=1000.0, missing_fields=[],
        ),
        "new_goal": new_goal,
        "new_goal_added": True,
        "extraction_llm_ok": True,
    })

    budget = BudgetAllocation(
        category=TransactionCategory.SHOPPING, allocated_amount=600.0,
        spent_amount=500.0, period_start=date.today().replace(day=1),
        period_end=date.today() + timedelta(days=30),
    )
    state = _make_state(
        goals=[], monthly_income=3000.0, budget_allocations=[budget],
        messages=[DummyMessage("I want to save 8000 by next year for a laptop.")],
    )
    result = _run_with_state(state)

    assert len(result["goals"]) == 1
    assert result["goals"][0].name == "Laptop Fund"
    assert result["pending_confirmation"]["action"] == "approve_goal_planning"


@patch("app.agents.goal_planning.agent.run_react_loop")
def test_agent_no_new_goal_still_evaluates(mock_react):
    mock_react.side_effect = _mock_react({
        "extraction": GoalExtractionResult(is_goal_intent=False, missing_fields=[]),
        "extraction_llm_ok": False,
    })

    goal = FinancialGoal(
        id="g4", name="Emergency Fund", target_amount=1200.0,
        current_amount=0.0, target_date=date.today() + timedelta(days=120),
    )
    state = _make_state(
        goals=[goal], monthly_income=2000.0, budget_allocations=[],
        messages=[DummyMessage("hello")],
    )
    result = _run_with_state(state)
    assert "goals" in result
    assert len(result["goals"]) >= 1
