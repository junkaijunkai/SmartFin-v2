"""deepeval -- Goal Extractor field extraction accuracy (L1)."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.agents.goal_planning.extractor import extract_goal_from_message

pytestmark = pytest.mark.eval

# expected_output uses the full GoalExtractionResult schema so the judge
# does not penalise for "extra fields" that are always present as null
_CASES = [
    (
        "I want to save GBP8,000 for a laptop by December 2026",
        '{"is_goal_intent":true,"name":"Laptop Fund","target_amount":8000.0,"target_date":"2026-12-31","current_amount":null,"missing_fields":[]}',
        "Goal intent true; name=Laptop Fund; target_amount=8000; target_date=2026-12-31; missing_fields empty",
    ),
    (
        "I'd like to start saving for a house deposit",
        '{"is_goal_intent":true,"name":"House Deposit","target_amount":null,"target_date":null,"current_amount":null,"missing_fields":["target_amount","target_date"]}',
        "Goal intent true; no amount or date stated; missing_fields contains target_amount and target_date",
    ),
    (
        "What's the weather like today?",
        '{"is_goal_intent":false,"name":null,"target_amount":null,"target_date":null,"current_amount":null,"missing_fields":[]}',
        "No financial goal intent; all other fields null; missing_fields empty",
    ),
]


@pytest.mark.parametrize("message,expected_json,description", _CASES)
def test_goal_extraction(message: str, expected_json: str, description: str, judge) -> None:
    result, succeeded = extract_goal_from_message(message)
    assert succeeded, "LLM goal extraction failed -- API may be unavailable"

    test_case = LLMTestCase(
        input=message,
        actual_output=result.model_dump_json(),
        expected_output=expected_json,
    )
    metric = GEval(
        name="GoalExtractionAccuracy",
        criteria=(
            "Evaluate whether actual_output correctly extracts the financial goal from input. "
            f"Key requirements: {description}. "
            "Focus only on: is_goal_intent, name, target_amount, target_date, missing_fields. "
            "The 'name' field: judge on semantic intent, not exact string. "
            "For example,'House Deposit' and 'House Deposit Fund' both mean the user wants to save for a house — either is correct. "
            "Ignore differences in null vs absent optional fields. "
            "Score 1.0 if is_goal_intent matches AND amount/date fields are correct AND name is semantically equivalent. "
            "Score 0.0 only if is_goal_intent is wrong or a clearly stated amount/date is missed."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.8,
        model=judge,
        async_mode=False,
    )
    assert_test(test_case, [metric])
