"""Capability eval: intent classifier routing accuracy (L3)."""

from __future__ import annotations

import pytest

from app.orchestrator.intent_classifier import classify_intent
from tests.evals.assertions import assert_equal
from tests.evals.loaders import GoldenCase, load_goldens
from tests.evals.reporting import record_result

pytestmark = pytest.mark.eval


@pytest.mark.parametrize("case", load_goldens("intent_routing"), ids=lambda c: c.id)
def test_intent_routing(case: GoldenCase) -> None:
    capability = "intent_routing"
    expected_agent = case.expected["agent"]
    try:
        actual = classify_intent(case.input)
        assert_equal(case.id, "agent", actual, expected_agent)
    except AssertionError as exc:
        record_result(case.id, capability, False, str(exc))
        raise
    record_result(case.id, capability, True)
