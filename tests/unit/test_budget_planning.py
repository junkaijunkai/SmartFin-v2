import pytest
from unittest.mock import MagicMock, patch
from datetime import date
from app.agents.budget_planning.extractor import extract_budget_request
from app.agents.budget_planning.planner import (
    generate_budget_allocations,
    calculate_monthly_spending,
    evaluate_budget_progress,
    generate_budget_warnings,
)
from app.agents.budget_planning.agent import budget_planning_node
from app.orchestrator.state_view import AgentStateView
from app.state import BudgetAllocation, TransactionCategory


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


# ---------------------------------------------------------------------------
# extractor.py tests
# ---------------------------------------------------------------------------

@patch("app.agents.budget_planning.extractor.ChatAnthropic")
def test_extract_budget_request_basic(mock_chat_anthropic):
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    mock_result = MagicMock()
    mock_result.user_message = "Help me plan my food and transport budget"
    mock_result.monthly_income = None
    mock_result.categories_requested = ["food", "transport"]
    mock_result.needs_clarification = False

    mock_structured.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_anthropic.return_value = mock_llm

    state = {
        "messages": [DummyMessage("Help me plan my food and transport budget")],
        "monthly_income": 5000,
    }

    result = extract_budget_request(state)

    assert result["intent"] == "budget_planning"
    assert result["monthly_income"] == 5000
    assert "food" in result["categories_requested"]
    assert "transport" in result["categories_requested"]
    assert result["needs_clarification"] is False


@patch("app.agents.budget_planning.extractor.ChatAnthropic")
def test_extract_budget_request_needs_clarification_when_income_missing(mock_chat_anthropic):
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    mock_result = MagicMock()
    mock_result.user_message = "Please help me plan my monthly budget"
    mock_result.monthly_income = None
    mock_result.categories_requested = []
    mock_result.needs_clarification = True

    mock_structured.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_anthropic.return_value = mock_llm

    state = {
        "messages": [DummyMessage("Please help me plan my monthly budget")],
    }

    result = extract_budget_request(state)

    assert result["intent"] == "budget_planning"
    assert result["monthly_income"] is None
    assert result["needs_clarification"] is True


@patch("app.agents.budget_planning.extractor.ChatAnthropic")
def test_extract_budget_request_empty_messages(mock_chat_anthropic):
    mock_llm = MagicMock()
    mock_structured = MagicMock()

    mock_result = MagicMock()
    mock_result.user_message = ""
    mock_result.monthly_income = None
    mock_result.categories_requested = []
    mock_result.needs_clarification = False

    mock_structured.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_anthropic.return_value = mock_llm

    state = {
        "messages": [],
        "monthly_income": 4000,
    }

    result = extract_budget_request(state)

    assert result["intent"] == "budget_planning"
    assert result["user_message"] == ""
    assert result["monthly_income"] == 4000
    assert result["categories_requested"] == []
    assert result["needs_clarification"] is False

# ---------------------------------------------------------------------------
# planner.py tests
# ---------------------------------------------------------------------------

def test_generate_budget_allocations_basic():
    category_monthly_avg = {
        "food": 500,
        "transport": 200,
        "housing": 1200,
    }
    category_trends = {
        "food": "stable",
        "transport": "rising",
        "housing": "fixed",
    }

    result = generate_budget_allocations(
        monthly_income=5000,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=None,
    )

    assert result["food"] == 525.00
    assert result["transport"] == 220.00
    assert result["housing"] == 1200.00


def test_generate_budget_allocations_keep_existing_budget():
    category_monthly_avg = {
        "food": 500,
        "transport": 200,
    }
    category_trends = {
        "food": "stable",
        "transport": "rising",
    }
    existing_budget = {
        "food": 600
    }

    result = generate_budget_allocations(
        monthly_income=5000,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=existing_budget,
    )

    assert result["food"] == 600.00
    assert result["transport"] == 220.00


def test_generate_budget_allocations_scale_down_when_exceed_income_limit():
    category_monthly_avg = {
        "housing": 2500,
        "food": 1000,
        "transport": 500,
    }
    category_trends = {
        "housing": "fixed",
        "food": "stable",
        "transport": "stable",
    }

    result = generate_budget_allocations(
        monthly_income=3000,
        category_monthly_avg=category_monthly_avg,
        category_trends=category_trends,
        existing_budget=None,
    )

    total_budget = sum(result.values())
    assert total_budget <= 2700.0 + 0.1


def test_calculate_monthly_spending_basic():
    transactions = [
        {"date": "2026-04-01", "category": "food", "amount": 20},
        {"date": "2026-04-02", "category": "food", "amount": 30.5},
        {"date": "2026-04-03", "category": "transport", "amount": 15},
    ]

    result = calculate_monthly_spending(transactions)

    assert result["food"] == 50.5
    assert result["transport"] == 15.0


def test_calculate_monthly_spending_ignore_invalid_and_non_positive_values():
    transactions = [
        {"date": "2026-04-01", "category": "food", "amount": 20},
        {"date": "2026-04-02", "category": "food", "amount": 0},
        {"date": "2026-04-03", "category": "food", "amount": -5},
        {"date": "2026-04-04", "category": "food", "amount": "invalid"},
        {"date": "2026-04-05", "amount": 10},
    ]

    result = calculate_monthly_spending(transactions)

    assert result["food"] == 20.0
    assert result["uncategorized"] == 10.0


def test_evaluate_budget_progress_on_track():
    budget_allocations = {
        "food": 600
    }
    actual_spending = {
        "food": 250
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=15,
        days_in_month=30,
    )

    assert result["food"]["spent"] == 250.0
    assert result["food"]["remaining"] == 350.0
    assert result["food"]["usage_ratio"] == round(250 / 600, 3)
    assert result["food"]["status"] == "on_track"


def test_evaluate_budget_progress_near_limit():
    budget_allocations = {
        "food": 600
    }
    actual_spending = {
        "food": 400
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=15,
        days_in_month=30,
    )

    assert result["food"]["status"] == "near_limit"


def test_evaluate_budget_progress_exceeded():
    budget_allocations = {
        "food": 600
    }
    actual_spending = {
        "food": 700
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=15,
        days_in_month=30,
    )

    assert result["food"]["status"] == "exceeded"
    assert result["food"]["remaining"] == -100.0


def test_evaluate_budget_progress_spending_without_budget():
    budget_allocations = {}
    actual_spending = {
        "misc": 100
    }

    result = evaluate_budget_progress(
        budget_allocations=budget_allocations,
        actual_spending=actual_spending,
        current_day=10,
        days_in_month=30,
    )

    assert result["misc"]["status"] == "exceeded"


def test_evaluate_budget_progress_invalid_days_in_month():
    with pytest.raises(ValueError):
        evaluate_budget_progress(
            budget_allocations={"food": 100},
            actual_spending={"food": 50},
            current_day=1,
            days_in_month=0,
        )


def test_generate_budget_warnings_low_medium_high():
    progress = {
        "food": {
            "usage_ratio": 0.82,
            "expected_ratio_by_today": 0.75,
            "status": "on_track",
        },
        "transport": {
            "usage_ratio": 0.75,
            "expected_ratio_by_today": 0.50,
            "status": "near_limit",
        },
        "entertainment": {
            "usage_ratio": 1.05,
            "expected_ratio_by_today": 0.50,
            "status": "exceeded",
        },
    }

    warnings = generate_budget_warnings(progress)

    assert len(warnings) == 3

    severity_map = {w["category"]: w["severity"] for w in warnings}

    assert severity_map["food"] == "low"
    assert severity_map["transport"] in ["medium", "high"]
    assert severity_map["entertainment"] == "high"


# ---------------------------------------------------------------------------
# agent.py tests  (mock run_react_loop, test pre/post processing)
# ---------------------------------------------------------------------------


def _run_budget(state: dict) -> dict:
    """Wrap state in AgentStateView and call the agent."""
    return budget_planning_node(AgentStateView(state, "budget_planning"))


@patch("app.agents.budget_planning.agent.run_react_loop")
def test_budget_planning_node_end_to_end(mock_react):
    """Test that the agent produces correct state when ReAct loop populates tool_ctx."""

    def side(tool_ctx=None, **kw):
        if tool_ctx is not None:
            tool_ctx["budget_request"] = {
                "intent": "budget_planning",
                "monthly_income": 5000,
                "categories_requested": ["food", "transport"],
                "needs_clarification": False,
            }
            tool_ctx["raw_allocations"] = {"food": 600.0, "transport": 200.0}
            tool_ctx["actual_spending"] = {"food": 130.0, "entertainment": 200.0, "housing": 1200.0, "transport": 60.0}
            tool_ctx["progress"] = {
                "food": {"spent": 130.0, "budget": 600.0, "status": "on_track"},
                "transport": {"spent": 60.0, "budget": 200.0, "status": "on_track"},
            }
            tool_ctx["warnings"] = []
        return MagicMock(content="Budget plan created for food and transport."), []

    mock_react.side_effect = side

    state = {
        "messages": [DummyMessage("Help me plan my monthly budget for food and transport")],
        "monthly_income": 5000,
        "categorised_transactions": [],
        "expense_analysis": {
            "category_monthly_avg": {"food": 600, "transport": 200, "housing": 1200, "entertainment": 250},
            "category_trends": {"food": "stable", "transport": "stable", "housing": "fixed", "entertainment": "rising"},
        },
        "current_date": "2026-04-16",
    }

    new_state = _run_budget(state)

    assert "budget_allocations" in new_state
    assert "budget_progress" in new_state
    assert "budget_warnings" in new_state
    assert "budget_summary" in new_state
    assert "budget_request" in new_state

    assert isinstance(new_state["budget_allocations"], list)
    assert all(isinstance(a, BudgetAllocation) for a in new_state["budget_allocations"])
    assert any(a.category == TransactionCategory.FOOD for a in new_state["budget_allocations"])

    assert "food" in new_state["budget_progress"]
    assert isinstance(new_state["budget_warnings"], list)
    assert isinstance(new_state["budget_summary"], str)
    assert new_state["budget_request"]["intent"] == "budget_planning"


@patch("app.agents.budget_planning.agent.run_react_loop")
def test_budget_planning_node_needs_clarification_returns_llm_response(mock_react):
    """When income is missing the LLM asks the user in its end_turn text — no HITL triggered."""

    def side(tool_ctx=None, **kw):
        if tool_ctx is not None:
            tool_ctx["budget_request"] = {
                "intent": "budget_planning",
                "monthly_income": None,
                "categories_requested": [],
                "needs_clarification": True,
            }
        return MagicMock(content="Could you share your monthly income so I can create your budget?"), []

    mock_react.side_effect = side

    state = {
        "messages": [DummyMessage("Help me plan my budget")],
        "monthly_income": None,
        "expense_analysis": {"category_monthly_avg": {}, "category_trends": {}},
        "current_date": "2026-04-16",
    }

    result = _run_budget(state)

    assert isinstance(result["budget_summary"], str)
    assert "income" in result["budget_summary"].lower()
    assert result["budget_warnings"] == []
    assert result.get("pending_confirmation") is None


@patch("app.agents.budget_planning.agent.run_react_loop")
def test_budget_planning_node_income_excluded_from_allocations(mock_react):

    def side(tool_ctx=None, **kw):
        if tool_ctx is not None:
            tool_ctx["budget_request"] = {
                "intent": "budget_planning",
                "monthly_income": 5000,
                "categories_requested": [],
                "needs_clarification": False,
            }
            tool_ctx["raw_allocations"] = {"food": 500.0}  # no "income" in raw_allocations
            tool_ctx["actual_spending"] = {"food": 200.0}
            tool_ctx["progress"] = {"food": {"spent": 200.0, "budget": 500.0, "status": "on_track"}}
            tool_ctx["warnings"] = []
        return MagicMock(), []

    mock_react.side_effect = side

    state = {
        "messages": [DummyMessage("Plan my budget")],
        "monthly_income": 5000,
        "categorised_transactions": [],
        "expense_analysis": {
            "category_monthly_avg": {"food": 500, "income": 3200},
            "category_trends": {"food": "stable", "income": "fixed"},
        },
        "current_date": "2026-04-16",
    }

    result = _run_budget(state)

    categories = [a.category for a in result["budget_allocations"]]
    assert TransactionCategory.INCOME not in categories
    assert TransactionCategory.FOOD in categories


# ---------------------------------------------------------------------------
# extractor.py fallback tests
# ---------------------------------------------------------------------------

@patch("app.agents.budget_planning.extractor.ChatAnthropic")
def test_extract_budget_request_llm_failure_falls_back_to_state_income(mock_chat_anthropic):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("API down")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_anthropic.return_value = mock_llm

    state = {
        "messages": [DummyMessage("Help me plan my budget")],
        "monthly_income": 4000,
    }

    result = extract_budget_request(state)

    assert result["monthly_income"] == 4000
    assert result["categories_requested"] == []
    assert result["needs_clarification"] is False


@patch("app.agents.budget_planning.extractor.ChatAnthropic")
def test_extract_budget_request_llm_failure_no_state_income_needs_clarification(mock_chat_anthropic):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("API down")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat_anthropic.return_value = mock_llm

    state = {
        "messages": [DummyMessage("Help me plan my budget")],
    }

    result = extract_budget_request(state)

    assert result["monthly_income"] is None
    assert result["needs_clarification"] is True