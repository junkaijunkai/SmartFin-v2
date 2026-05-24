"""deepeval — Expense Categoriser category accuracy (L1)."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.agents.expense_analysis.categoriser import categorise_transactions
from tests.evals.conftest import _txn

pytestmark = pytest.mark.eval

_TRANSACTIONS = [
    _txn("e1", "Landlord", "Monthly rent payment", 1200.0),
    _txn("e2", "Netflix", "Monthly streaming subscription", 15.99),
    _txn("e3", "Tesco", "Weekly grocery shopping", 78.40),
    _txn("e4", "Company Ltd", "Monthly salary", -3200.0),
    _txn("e5", "Uber", "Taxi ride to airport", 32.50),
]

_EXPECTED = {
    "e1": "housing",
    "e2": "entertainment",
    "e3": "food",
    "e4": "income",
    "e5": "transport",
}


def test_expense_categorisation(judge) -> None:
    categorised, succeeded = categorise_transactions(_TRANSACTIONS)
    assert succeeded, "LLM categorisation fell back to keyword rules — API may be unavailable"

    for txn in categorised:
        expected_cat = _EXPECTED[txn.id]
        test_case = LLMTestCase(
            input=f"merchant={txn.merchant} | description={txn.description} | amount={txn.amount:.2f}",
            actual_output=txn.category.value,
            expected_output=expected_cat,
        )
        metric = GEval(
            name="CategoryAccuracy",
            criteria=(
                "The actual_output category must correctly classify the transaction described in input. "
                "Score 1.0 if it matches the expected_output; 0.0 if it is a clearly wrong category."
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
