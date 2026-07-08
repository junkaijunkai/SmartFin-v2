"""Capability eval: expense categorisation accuracy (L1)."""

from __future__ import annotations

import pytest

from app.agents.expense_analysis.categoriser import categorise_transactions
from app.state import TransactionCategory
from tests.evals.assertions import assert_equal
from tests.evals.conftest import _txn
from tests.evals.loaders import GoldenCase, load_goldens
from tests.evals.reporting import record_result

pytestmark = pytest.mark.eval


@pytest.mark.parametrize("case", load_goldens("expense_categorisation"), ids=lambda c: c.id)
def test_expense_categorisation(case: GoldenCase) -> None:
    capability = "expense_categorisation"
    expected_category = case.expected["category"]
    txn = _txn(
        case.id,
        case.context["merchant"],
        case.input,
        case.context["amount"],
        TransactionCategory.OTHER,
        days_ago=case.context.get("days_ago", 5),
    )

    try:
        categorised, succeeded = categorise_transactions([txn])
        assert succeeded, f"{case.id}: LLM categorisation failed and used fallback"
        assert_equal(case.id, "category", categorised[0].category.value, expected_category)
    except AssertionError as exc:
        record_result(case.id, capability, False, str(exc))
        raise
    record_result(case.id, capability, True)
