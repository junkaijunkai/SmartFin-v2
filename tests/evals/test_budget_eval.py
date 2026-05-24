"""deepeval -- Budget Request Extractor field extraction accuracy (L1)."""

from __future__ import annotations

import json
import pytest
from langchain_core.messages import HumanMessage
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.agents.budget_planning.extractor import extract_budget_request

pytestmark = pytest.mark.eval

_CASES = [
    (
        # "budget of GBP4000" is a spending target, not income; extractor returns monthly_income=null
        "Set me a monthly budget of GBP4000, with GBP800 for food and GBP200 for transport",
        None,
        '{"monthly_income": null, "categories_requested": ["food", "transport"], "needs_clarification": true}',
        "No income stated in message or state. categories_requested=['food','transport']. needs_clarification=true.",
        ["User state: monthly_income is not set (null). No income was provided by the user."],
    ),
    (
        # No income mentioned; extractor must fall back to state_income=3500
        "Create a budget plan for me",
        3500.0,
        '{"monthly_income": 3500.0, "categories_requested": [], "needs_clarification": false}',
        "Income comes from application state (3500), not the message. needs_clarification=false because income is known.",
        ["User state: monthly_income=3500.0 was pre-populated from the user profile before this message."],
    ),
]


@pytest.mark.parametrize("message,state_income,expected_json,description,context", _CASES)
def test_budget_extraction(
    message: str,
    state_income: float | None,
    expected_json: str,
    description: str,
    context: list[str],
    judge,
) -> None:
    state = {
        "messages": [HumanMessage(content=message)],
        "monthly_income": state_income,
    }
    result = extract_budget_request(state)

    test_case = LLMTestCase(
        input=message,
        actual_output=json.dumps(result),
        expected_output=expected_json,
        context=context,
    )
    metric = GEval(
        name="BudgetExtractionAccuracy",
        criteria=(
            "Given the input message and context (which describes the application state), "
            "evaluate whether actual_output (JSON) matches expected_output. "
            f"Correct behaviour: {description} "
            "Score 1.0 if categories_requested, monthly_income, and needs_clarification "
            "match expected_output. Score 0.0 if any clearly contradicts it."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.8,
        model=judge,
        async_mode=False,
    )
    assert_test(test_case, [metric])
