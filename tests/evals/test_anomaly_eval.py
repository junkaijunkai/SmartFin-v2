"""Capability eval: anomaly explanation quality (L2)."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.agents.anomaly_detection.extractor import extract_and_detect
from app.state import TransactionCategory
from tests.evals.conftest import _txn
from tests.evals.loaders import GoldenCase, load_goldens
from tests.evals.reporting import record_result

pytestmark = pytest.mark.eval


def _transactions(case: GoldenCase):
    return [
        _txn(
            txn["id"],
            txn["merchant"],
            txn["description"],
            txn["amount"],
            TransactionCategory(txn["category"]),
            days_ago=txn.get("days_ago", 5),
        )
        for txn in case.context["transactions"]
    ]


@pytest.mark.parametrize("case", load_goldens("anomaly_detection"), ids=lambda c: c.id)
def test_anomaly_explanation_quality(case: GoldenCase, judge) -> None:
    capability = "anomaly_detection"
    try:
        flags, explanation_text = extract_and_detect([], _transactions(case))
        assert flags, f"{case.id}: no anomaly flags detected"
        test_case = LLMTestCase(
            input=case.input,
            actual_output=explanation_text,
            retrieval_context=case.context["retrieval_context"],
        )
        metrics = [
            AnswerRelevancyMetric(threshold=case.expected.get("answer_relevancy_threshold", 0.65), model=judge, async_mode=False),
            FaithfulnessMetric(threshold=case.expected.get("faithfulness_threshold", 0.65), model=judge, async_mode=False),
        ]
        assert_test(test_case, metrics)
    except AssertionError as exc:
        record_result(case.id, capability, False, str(exc))
        raise
    record_result(case.id, capability, True)
