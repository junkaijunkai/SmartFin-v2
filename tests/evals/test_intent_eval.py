"""deepeval — Intent Classifier routing accuracy (L3)."""

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.orchestrator.intent_classifier import classify_intent

pytestmark = pytest.mark.eval

_CASES = [
    ("Show me a breakdown of my spending last month", "expense_analysis"),
    ("Set me a monthly budget of £3000, split between food and transport", "budget_planning"),
    ("I want to save £10,000 for a house deposit by end of 2026", "goal_planning"),
    ("There's a suspicious £500 charge on my account I don't recognise", "anomaly_detection"),
    ("What's my overall financial health looking like right now?", "health_assessment"),
]


@pytest.mark.parametrize("message,expected_agent", _CASES)
def test_intent_routing(message: str, expected_agent: str, judge) -> None:
    actual = classify_intent(message)
    test_case = LLMTestCase(
        input=message,
        actual_output=actual,
        expected_output=expected_agent,
    )
    metric = GEval(
        name="IntentRoutingAccuracy",
        criteria=(
            "The actual_output must be the same agent name as expected_output. "
            "Score 1.0 only if they match exactly; 0.0 otherwise."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.9,
        model=judge,
        async_mode=False,
    )
    assert_test(test_case, [metric])
