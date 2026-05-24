"""deepeval — Health Advisory observation quality (L2).

Scenario: POOR health rating.
  - Monthly income: £3,000
  - Rent: £2,100 → DTI = 70% (> DTI_POOR 50%)
  - Savings: £400 → reserve_months = 400/2100 ≈ 0.19 (< RESERVE_FAIR 1.0)
  Expected: advisory observations address high DTI and critically low reserves.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.agents.health_assessment.assessor import assess_health
from app.state import SpendingTrend, TransactionCategory
from tests.evals.conftest import _txn

pytestmark = pytest.mark.eval

_MONTHLY_INCOME = 3000.0

_TRANSACTIONS = [
    # Housing (within last 30 days for DTI computation)
    _txn("h1", "Landlord", "Monthly rent", 2100.0, TransactionCategory.HOUSING, days_ago=5),
    # Savings deposit (for reserve_months computation)
    _txn("h2", "ISA Account", "Monthly savings transfer", 400.0, TransactionCategory.SAVINGS, days_ago=3),
    # Income (negative = income)
    _txn("h3", "Employer Ltd", "Monthly salary", -3000.0, TransactionCategory.INCOME, days_ago=1),
]

_SPENDING_TRENDS = [
    SpendingTrend(
        category=TransactionCategory.HOUSING,
        current_period_total=2100.0,
        previous_period_total=2100.0,
        deviation_pct=0.0,
    ),
    SpendingTrend(
        category=TransactionCategory.FOOD,
        current_period_total=450.0,
        previous_period_total=320.0,
        deviation_pct=40.6,
    ),
]

_CONTEXT = [
    "Financial health rating: POOR",
    "Debt-to-income ratio: 70% (threshold for poor: 50%)",
    "Liquid reserve months: 0.19 (critically low; target: 3 months)",
    "Income concentration risk: No",
    "Sustained overspending: No",
    "Monthly income: £3000",
    "Top spending trend — housing: £2100/month (+0.0% vs prior period)",
    "Top spending trend — food: £450/month (+40.6% vs prior period)",
]


def test_health_advisory_relevancy(judge) -> None:
    summary, _ = assess_health(_TRANSACTIONS, _MONTHLY_INCOME, _SPENDING_TRENDS)
    observations_text = "\n".join(summary.observations)

    test_case = LLMTestCase(
        input="Provide personalised financial health observations based on the metrics",
        actual_output=observations_text,
        retrieval_context=_CONTEXT,
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False)
    assert_test(test_case, [metric])


def test_health_advisory_faithfulness(judge) -> None:
    summary, _ = assess_health(_TRANSACTIONS, _MONTHLY_INCOME, _SPENDING_TRENDS)
    observations_text = "\n".join(summary.observations)

    test_case = LLMTestCase(
        input="Provide personalised financial health observations based on the metrics",
        actual_output=observations_text,
        retrieval_context=_CONTEXT,
    )
    metric = FaithfulnessMetric(threshold=0.5, model=judge, async_mode=False)
    assert_test(test_case, [metric])
