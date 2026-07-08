"""Capability eval: health advisory observation quality (L2)."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.agents.health_assessment.assessor import assess_health
from app.state import SpendingTrend, TransactionCategory
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


def _trends(case: GoldenCase) -> list[SpendingTrend]:
    return [
        SpendingTrend(
            category=TransactionCategory(trend["category"]),
            current_period_total=trend["current_period_total"],
            previous_period_total=trend.get("previous_period_total"),
            deviation_pct=trend.get("deviation_pct"),
        )
        for trend in case.context.get("spending_trends", [])
    ]


@pytest.mark.parametrize("case", load_goldens("health_assessment"), ids=lambda c: c.id)
def test_health_advisory_quality(case: GoldenCase, judge) -> None:
    capability = "health_assessment"
    try:
        summary, _ = assess_health(
            _transactions(case),
            case.context["monthly_income"],
            _trends(case),
        )
        observations_text = "\n".join(summary.observations)
        test_case = LLMTestCase(
            input=case.input,
            actual_output=observations_text,
            retrieval_context=case.context["retrieval_context"],
        )
        metrics = [
            AnswerRelevancyMetric(threshold=case.expected.get("answer_relevancy_threshold", 0.65), model=judge, async_mode=False),
            FaithfulnessMetric(threshold=case.expected.get("faithfulness_threshold", 0.6), model=judge, async_mode=False),
        ]
        assert_test(test_case, metrics)
    except AssertionError as exc:
        record_result(case.id, capability, False, str(exc))
        raise
    record_result(case.id, capability, True)
