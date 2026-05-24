"""deepeval — Anomaly Explainer explanation quality (L2).

Transactions are crafted so the IQR detector flags one outlier:
  food amounts [20, 25, 22, 18, 500] → upper fence = Q3 + 1.5*IQR = 25 + 1.5*5 = 32.5
  £500 >> 32.5 → flagged as UNUSUAL_AMOUNT.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.agents.anomaly_detection.extractor import extract_and_detect
from app.state import TransactionCategory
from tests.evals.conftest import _txn

pytestmark = pytest.mark.eval

_FOOD_TRANSACTIONS = [
    _txn("f1", "Tesco", "Weekly groceries", 20.0, TransactionCategory.FOOD, days_ago=28),
    _txn("f2", "Sainsbury's", "Weekly groceries", 25.0, TransactionCategory.FOOD, days_ago=21),
    _txn("f3", "Lidl", "Weekly groceries", 22.0, TransactionCategory.FOOD, days_ago=14),
    _txn("f4", "Aldi", "Weekly groceries", 18.0, TransactionCategory.FOOD, days_ago=7),
    _txn("f5", "Gourmet Restaurant", "Birthday dinner for 12 people", 500.0, TransactionCategory.FOOD, days_ago=2),
]

_CONTEXT = [
    "Transaction: Gourmet Restaurant, £500.00, food category",
    "Statistical reason: Amount 500.00 exceeds the IQR upper fence of approximately £32.50 for the food category",
    "Typical food spending for this account: £18–£25 per transaction",
]


def test_anomaly_explanation_relevancy(judge) -> None:
    flags, explanation_text = extract_and_detect([], _FOOD_TRANSACTIONS)
    assert flags, "No anomaly flags detected — check transaction construction"

    test_case = LLMTestCase(
        input="Explain the flagged anomalous food transactions for this account",
        actual_output=explanation_text,
        retrieval_context=_CONTEXT,
    )
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge, async_mode=False)
    assert_test(test_case, [metric])


def test_anomaly_explanation_faithfulness(judge) -> None:
    flags, explanation_text = extract_and_detect([], _FOOD_TRANSACTIONS)
    assert flags, "No anomaly flags detected — check transaction construction"

    test_case = LLMTestCase(
        input="Explain the flagged anomalous food transactions for this account",
        actual_output=explanation_text,
        retrieval_context=_CONTEXT,
    )
    metric = FaithfulnessMetric(threshold=0.7, model=judge, async_mode=False)
    assert_test(test_case, [metric])
