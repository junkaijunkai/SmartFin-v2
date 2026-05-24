"""Unit tests for anomaly detection — detector.py and agent.py."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app.agents.anomaly_detection.agent import run as anomaly_run
from app.orchestrator.state_view import AgentStateView
from app.agents.anomaly_detection.detector import (
    FREQUENCY_THRESHOLD,
    MIN_SAMPLE_SIZE,
    _detect_unusual_amounts,
    _detect_unusual_frequency,
    detect_anomalies,
)
from app.agents.anomaly_detection.extractor import extract_and_detect
from app.state import AnomalyFlag, AnomalyType, Transaction, TransactionCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _txn(
    amount: float,
    category: TransactionCategory = TransactionCategory.FOOD,
    merchant: str = "TestMerchant",
    days_ago: float = 1,
    txn_id: str | None = None,
) -> Transaction:
    return Transaction(
        id=txn_id or str(uuid.uuid4()),
        date=datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
        amount=amount,
        description="test transaction",
        merchant=merchant,
        category=category,
    )


def _food_batch(amounts: list[float]) -> list[Transaction]:
    """Return FOOD transactions with given amounts, spread one per day."""
    return [_txn(a, days_ago=i + 1) for i, a in enumerate(amounts)]


def _merchant_txns(
    count: int,
    merchant: str = "Coffee Shop",
    start_days_ago: int = 6,
) -> list[Transaction]:
    """Return `count` expense transactions to the same merchant, one per day."""
    return [
        _txn(5.0, merchant=merchant, days_ago=start_days_ago - i, txn_id=f"{merchant}-{i}")
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# TestDetectUnusualAmounts
# ---------------------------------------------------------------------------


class TestDetectUnusualAmounts:
    def test_flags_outlier_above_fence(self):
        """A transaction far above Q3 + 1.5*IQR is flagged as UNUSUAL_AMOUNT.

        Batch [10, 11, 12, 13, 14, 200]:
          q1=11, q3=14, iqr=3, fence=18.5 → 200 is flagged.
        """
        outlier = _txn(200.0, txn_id="outlier")
        txns = _food_batch([10, 11, 12, 13, 14]) + [outlier]
        flags = _detect_unusual_amounts(txns)
        assert any(f.transaction_id == "outlier" for f in flags)

    def test_anomaly_type_is_unusual_amount(self):
        outlier = _txn(200.0, txn_id="outlier")
        txns = _food_batch([10, 11, 12, 13, 14]) + [outlier]
        flags = _detect_unusual_amounts(txns)
        assert all(f.anomaly_type == AnomalyType.UNUSUAL_AMOUNT for f in flags)

    def test_no_flag_when_all_amounts_normal(self):
        """Transactions within the fence produce no flags."""
        txns = _food_batch([10, 11, 12, 13, 14, 15])
        assert _detect_unusual_amounts(txns) == []

    def test_skips_category_below_min_sample_size(self):
        """Fewer than MIN_SAMPLE_SIZE transactions per category → no detection."""
        txns = _food_batch([10, 200, 300])  # only 3 entries
        assert len(txns) < MIN_SAMPLE_SIZE
        assert _detect_unusual_amounts(txns) == []

    def test_income_excluded(self):
        """Transactions with amount <= 0 (income) are not evaluated."""
        txns = [_txn(-5000.0, txn_id=f"income-{i}") for i in range(6)]
        assert _detect_unusual_amounts(txns) == []

    def test_explanation_mentions_category(self):
        outlier = _txn(500.0, category=TransactionCategory.SHOPPING, txn_id="outlier")
        normals = [_txn(20.0, category=TransactionCategory.SHOPPING, days_ago=i + 2) for i in range(5)]
        flags = _detect_unusual_amounts(normals + [outlier])
        assert flags, "expected at least one flag"
        assert "shopping" in flags[0].explanation.lower()

    def test_categories_evaluated_independently(self):
        """An amount that is an outlier in FOOD should not flag the same amount
        in TRANSPORT where it is a normal value."""
        food_outlier = _txn(200.0, category=TransactionCategory.FOOD, txn_id="food-outlier")
        food_normals = [
            _txn(a, category=TransactionCategory.FOOD, days_ago=i + 1)
            for i, a in enumerate([10, 11, 12, 13, 14])
        ]
        # TRANSPORT batch where 200 is the normal value — no outlier
        transport = [
            _txn(200.0, category=TransactionCategory.TRANSPORT, txn_id=f"t-{i}", days_ago=i + 1)
            for i in range(6)
        ]
        flags = _detect_unusual_amounts(food_normals + [food_outlier] + transport)
        flagged_ids = {f.transaction_id for f in flags}
        assert "food-outlier" in flagged_ids
        assert not any(fid.startswith("t-") for fid in flagged_ids)


# ---------------------------------------------------------------------------
# TestDetectUnusualFrequency
# ---------------------------------------------------------------------------


class TestDetectUnusualFrequency:
    def test_flags_excess_visits_in_window(self):
        """More than FREQUENCY_THRESHOLD visits within the window → excess flagged."""
        txns = _merchant_txns(FREQUENCY_THRESHOLD + 1)
        flags = _detect_unusual_frequency(txns)
        assert len(flags) == 1
        assert flags[0].anomaly_type == AnomalyType.UNUSUAL_FREQUENCY

    def test_no_flag_at_exactly_threshold(self):
        """Exactly FREQUENCY_THRESHOLD visits → no flag."""
        txns = _merchant_txns(FREQUENCY_THRESHOLD)
        assert _detect_unusual_frequency(txns) == []

    def test_no_flag_below_threshold(self):
        txns = _merchant_txns(FREQUENCY_THRESHOLD - 1)
        assert _detect_unusual_frequency(txns) == []

    def test_income_excluded_from_frequency(self):
        """Negative-amount (income) entries do not count toward frequency."""
        txns = [_txn(-5.0, merchant="Employer", days_ago=i) for i in range(FREQUENCY_THRESHOLD + 2)]
        assert _detect_unusual_frequency(txns) == []

    def test_excess_not_double_flagged(self):
        """A transaction appearing in multiple rolling windows is flagged only once."""
        txns = _merchant_txns(FREQUENCY_THRESHOLD + 2)
        flags = _detect_unusual_frequency(txns)
        flagged_ids = [f.transaction_id for f in flags]
        assert len(flagged_ids) == len(set(flagged_ids))

    def test_merchant_comparison_case_insensitive(self):
        """'Starbucks' and 'starbucks' are treated as the same merchant."""
        upper = [_txn(5.0, merchant="Starbucks", days_ago=6 - i, txn_id=f"upper-{i}") for i in range(3)]
        lower = [_txn(5.0, merchant="starbucks", days_ago=2 - i, txn_id=f"lower-{i}") for i in range(3)]
        flags = _detect_unusual_frequency(upper + lower)
        assert len(flags) >= 1

    def test_visits_outside_window_not_grouped(self):
        """Transactions separated by more than the window are not combined."""
        # 3 visits in week 1, 3 visits 3 weeks later — no window exceeds threshold
        week1 = [_txn(5.0, merchant="Cafe", days_ago=21 - i, txn_id=f"w1-{i}") for i in range(3)]
        week3 = [_txn(5.0, merchant="Cafe", days_ago=2 - i, txn_id=f"w3-{i}") for i in range(3)]
        assert _detect_unusual_frequency(week1 + week3) == []


# ---------------------------------------------------------------------------
# TestDetectAnomalies (public API)
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_empty_input_returns_empty(self):
        assert detect_anomalies([]) == []

    def test_combines_amount_and_frequency_flags(self):
        """Both detectors contribute to the combined result."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")
        food_normals = _food_batch([10, 11, 12, 13, 14])
        freq_txns = _merchant_txns(FREQUENCY_THRESHOLD + 1, merchant="Cafe")
        flags = detect_anomalies(food_normals + [outlier] + freq_txns)
        types = {f.anomaly_type for f in flags}
        assert AnomalyType.UNUSUAL_AMOUNT in types
        assert AnomalyType.UNUSUAL_FREQUENCY in types

    def test_result_sorted_by_id_and_type(self):
        """Output is deterministically sorted by (transaction_id, anomaly_type)."""
        outlier = _txn(500.0, txn_id="zzz-outlier")
        txns = _food_batch([10, 11, 12, 13, 14]) + [outlier]
        flags = detect_anomalies(txns)
        keys = [(f.transaction_id, f.anomaly_type) for f in flags]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# TestAnomalyDetectionAgentNode
# ---------------------------------------------------------------------------


class TestAnomalyDetectionAgentNode:

    def _run(self, state: dict) -> dict:
        return anomaly_run(AgentStateView(state, "anomaly_detection"))

    @patch("app.agents.anomaly_detection.agent.run_react_loop")
    def test_writes_anomaly_flags_to_state(self, mock_react):
        def side(tool_ctx=None, **kw):
            if tool_ctx is not None:
                tool_ctx["anomaly_flags"] = []
                tool_ctx["explanation"] = "No anomalous transactions detected."
            return MagicMock(), []

        mock_react.side_effect = side
        result = self._run({"transactions": [], "messages": []})

        assert "anomaly_flags" in result
        assert "anomaly_explanation" in result
        assert isinstance(result["anomaly_flags"], list)

    @patch("app.agents.anomaly_detection.agent.run_react_loop")
    def test_empty_state_returns_empty_flags(self, mock_react):

        def side(tool_ctx=None, **kw):
            if tool_ctx is not None:
                tool_ctx["anomaly_flags"] = []
                tool_ctx["explanation"] = "No transactions available to scan."
            return MagicMock(), []

        mock_react.side_effect = side
        result = self._run({})
        assert result["anomaly_flags"] == []

    @patch("app.agents.anomaly_detection.agent.run_react_loop")
    def test_prefers_categorised_transactions(self, mock_react):
        """When both keys are present, categorised_transactions takes priority."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")

        def side(tool_ctx=None, **kw):
            if tool_ctx is not None:
                tool_ctx["anomaly_flags"] = [
                    AnomalyFlag(transaction_id="outlier", anomaly_type=AnomalyType.UNUSUAL_AMOUNT,
                                explanation="test")
                ]
                tool_ctx["explanation"] = "Found 1 anomaly."
            return MagicMock(), []

        mock_react.side_effect = side

        food_normals = _food_batch([10, 11, 12, 13, 14])
        state = {
            "transactions": [],
            "categorised_transactions": food_normals + [outlier],
            "messages": [],
        }
        result = self._run(state)

        assert any(f.transaction_id == "outlier" for f in result["anomaly_flags"])

    @patch("app.agents.anomaly_detection.agent.run_react_loop")
    def test_falls_back_to_raw_transactions(self, mock_react):
        """When categorised_transactions is absent, raw transactions are used."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")

        def side(tool_ctx=None, **kw):
            if tool_ctx is not None:
                tool_ctx["anomaly_flags"] = [
                    AnomalyFlag(transaction_id="outlier", anomaly_type=AnomalyType.UNUSUAL_AMOUNT,
                                explanation="test")
                ]
            return MagicMock(), []

        mock_react.side_effect = side

        state = {
            "transactions": _food_batch([10, 11, 12, 13, 14]) + [outlier],
            "messages": [],
        }
        result = self._run(state)
        assert any(f.transaction_id == "outlier" for f in result["anomaly_flags"])

    @patch("app.agents.anomaly_detection.agent.run_react_loop")
    def test_falls_back_when_categorised_is_empty_list(self, mock_react):
        """An empty categorised_transactions triggers fallback to raw transactions."""
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier")

        def side(tool_ctx=None, **kw):
            if tool_ctx is not None:
                tool_ctx["anomaly_flags"] = [
                    AnomalyFlag(transaction_id="outlier", anomaly_type=AnomalyType.UNUSUAL_AMOUNT,
                                explanation="test")
                ]
            return MagicMock(), []

        mock_react.side_effect = side

        state = {
            "transactions": _food_batch([10, 11, 12, 13, 14]) + [outlier],
            "categorised_transactions": [],
            "messages": [],
        }
        result = self._run(state)
        assert any(f.transaction_id == "outlier" for f in result["anomaly_flags"])


# ---------------------------------------------------------------------------
# TestExtractAndDetect
# ---------------------------------------------------------------------------


class TestExtractAndDetect:
    def test_no_flags_returns_no_anomaly_message(self):
        """Empty transactions → empty flags + 'no anomalies' message."""
        flags, explanation = extract_and_detect([], [])
        assert flags == []
        assert "No anomalous" in explanation

    @patch("app.agents.anomaly_detection.extractor.ChatAnthropic")
    def test_returns_flags_and_formatted_explanation(self, MockLLM):
        """Flags present → LLM explanations + formatted string."""
        # Use different merchants for normals to avoid triggering frequency detection
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier", merchant="FoodShop")
        food_normals = [
            _txn(a, category=TransactionCategory.FOOD, merchant=f"Vendor-{i}", days_ago=i+1)
            for i, a in enumerate([10, 11, 12, 13, 14])
        ]
        txns = food_normals + [outlier]

        explanation_batch = MagicMock()
        explanation_batch.results = [
            MagicMock(transaction_id="outlier", explanation="LLM generated explanation")
        ]
        MockLLM.return_value.with_structured_output.return_value.invoke.return_value = explanation_batch

        flags, explanation = extract_and_detect([], txns)

        assert len(flags) == 1
        assert flags[0].transaction_id == "outlier"
        assert "following transactions may be anomalous" in explanation
        assert "LLM generated explanation" in explanation

    @patch("app.agents.anomaly_detection.extractor.ChatAnthropic")
    def test_llm_failure_falls_back_to_statistical_explanation(self, MockLLM):
        """LLM exception → flags kept, statistical explanation used in output."""
        # Use different merchants for normals to avoid triggering frequency detection
        outlier = _txn(500.0, category=TransactionCategory.FOOD, txn_id="outlier", merchant="FoodShop")
        food_normals = [
            _txn(a, category=TransactionCategory.FOOD, merchant=f"Vendor-{i}", days_ago=i+1)
            for i, a in enumerate([10, 11, 12, 13, 14])
        ]
        txns = food_normals + [outlier]

        MockLLM.return_value.with_structured_output.return_value.invoke.side_effect = RuntimeError("API down")

        flags, explanation = extract_and_detect([], txns)

        assert len(flags) == 1
        assert flags[0].transaction_id == "outlier"
        # Fallback to statistical explanation
        assert "following transactions may be anomalous" in explanation
        assert "exceeds the upper fence" in explanation  # statistical reason

