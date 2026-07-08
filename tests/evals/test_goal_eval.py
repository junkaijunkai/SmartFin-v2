"""Capability eval: goal extraction field accuracy (L1)."""

from __future__ import annotations

from datetime import date

import pytest

from app.agents.goal_planning.extractor import extract_goal_from_message
from tests.evals.assertions import (
    assert_date_equal,
    assert_equal,
    assert_float_equal,
    assert_semantic_name,
    assert_set_equal,
)
from tests.evals.loaders import GoldenCase, load_goldens
from tests.evals.reporting import record_result

pytestmark = pytest.mark.eval


@pytest.mark.parametrize("case", load_goldens("goal_extraction"), ids=lambda c: c.id)
def test_goal_extraction(case: GoldenCase) -> None:
    capability = "goal_extraction"
    expected = case.expected
    today = date.fromisoformat(case.context.get("today", "2026-07-05"))

    try:
        result, succeeded = extract_goal_from_message(case.input, today=today)
        assert succeeded, f"{case.id}: LLM goal extraction failed and used fallback"
        assert_equal(case.id, "is_goal_intent", result.is_goal_intent, expected["is_goal_intent"])
        assert_equal(case.id, "is_update_intent", result.is_update_intent, expected.get("is_update_intent", False))
        assert_semantic_name(case.id, result.name, expected.get("name"), expected.get("name_aliases", []))
        assert_float_equal(case.id, "target_amount", result.target_amount, expected.get("target_amount"))
        assert_date_equal(case.id, "target_date", result.target_date, expected.get("target_date"))
        assert_float_equal(case.id, "current_amount", result.current_amount, expected.get("current_amount"))
        assert_set_equal(case.id, "missing_fields", result.missing_fields, expected.get("missing_fields", []))
    except AssertionError as exc:
        record_result(case.id, capability, False, str(exc))
        raise
    record_result(case.id, capability, True)
