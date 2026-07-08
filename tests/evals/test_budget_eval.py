"""Capability eval: budget request extraction field accuracy (L1)."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agents.budget_planning.extractor import extract_budget_request
from tests.evals.assertions import assert_equal, assert_float_equal, assert_set_equal
from tests.evals.loaders import GoldenCase, load_goldens
from tests.evals.reporting import record_result

pytestmark = pytest.mark.eval


@pytest.mark.parametrize("case", load_goldens("budget_extraction"), ids=lambda c: c.id)
def test_budget_extraction(case: GoldenCase) -> None:
    capability = "budget_extraction"
    expected = case.expected
    state = {
        "messages": [HumanMessage(content=case.input)],
        "monthly_income": case.context.get("monthly_income"),
    }

    try:
        result = extract_budget_request(state, context=case.context.get("conversation"))
        assert_equal(case.id, "intent", result["intent"], "budget_planning")
        assert_float_equal(case.id, "monthly_income", result["monthly_income"], expected.get("monthly_income"))
        assert_set_equal(case.id, "categories_requested", result["categories_requested"], expected.get("categories_requested", []))
        assert_equal(case.id, "needs_clarification", result["needs_clarification"], expected["needs_clarification"])
    except AssertionError as exc:
        record_result(case.id, capability, False, str(exc))
        raise
    record_result(case.id, capability, True)
