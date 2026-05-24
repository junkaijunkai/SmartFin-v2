"""
Transaction anomaly detector — pure statistical detection.

Two detection methods (no LLM decision-making):
  1. UNUSUAL_AMOUNT     — per-category IQR outlier (requires >= MIN_SAMPLE_SIZE transactions).
  2. UNUSUAL_FREQUENCY  — same merchant appearing > FREQUENCY_THRESHOLD times within
                          a rolling FREQUENCY_WINDOW_DAYS window.

LLM-based explanation generation is delegated to extractor.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta, timezone

from app.state import AnomalyFlag, AnomalyType, Transaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MIN_SAMPLE_SIZE = 4       # minimum transactions per category to run IQR detection
IQR_FACTOR = 1.5          # standard Tukey fence multiplier
FREQUENCY_WINDOW_DAYS = 7
FREQUENCY_THRESHOLD = 5   # more than this many visits to same merchant in window = flag


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iqr_upper_fence(values: list[float]) -> float:
    """Return the upper Tukey fence: Q3 + IQR_FACTOR * IQR."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Use integer indexing — avoids float interpolation for simplicity
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[(3 * n) // 4]
    iqr = q3 - q1
    return q3 + IQR_FACTOR * iqr


# 异常金额检测：根据Upper Tukey Fence检测
def _detect_unusual_amounts(transactions: list[Transaction]) -> list[AnomalyFlag]:
    """
    Flag transactions whose amount is an outlier within their category.

    Only expense transactions (amount > 0) are considered.
    Categories with fewer than MIN_SAMPLE_SIZE data points are skipped.
    """

    by_category: dict[str, list[Transaction]] = defaultdict(list)
    for t in transactions:
        if t.amount > 0:
            by_category[t.category].append(t)

    flags: list[AnomalyFlag] = []
    for category, txns in by_category.items():
        if len(txns) < MIN_SAMPLE_SIZE:
            # 交易样本到达一定量才进行统计分析
            continue

        amounts = [t.amount for t in txns]
        # 计算异常阈值
        fence = _iqr_upper_fence(amounts)
        category_mean = sum(amounts) / len(amounts)

        for t in txns:
            if t.amount > fence:
                flags.append(
                    AnomalyFlag(
                        transaction_id=t.id,
                        anomaly_type=AnomalyType.UNUSUAL_AMOUNT,
                        explanation=(
                            f"Amount {t.amount:.2f} exceeds the upper fence "
                            f"{fence:.2f} for category '{category}' "
                            f"(category average: {category_mean:.2f})."
                        ),
                    )
                )

    return flags

# 异常频率检测：同一商户在滚动窗口内出现过多
def _detect_unusual_frequency(transactions: list[Transaction]) -> list[AnomalyFlag]:
    """
    Flag transactions where the same merchant appears more than FREQUENCY_THRESHOLD
    times within any FREQUENCY_WINDOW_DAYS rolling window.

    Only expense transactions are considered. When flagged, only the excess
    transactions (beyond the threshold) are flagged to avoid duplicates.
    """
    # Normalise all datetimes to UTC-aware before comparison
    expense_txns = []
    for t in transactions:
        if t.amount <= 0:
            continue
        tx_dt = t.date
        if tx_dt.tzinfo is None:
            tx_dt = tx_dt.replace(tzinfo=timezone.utc)
        expense_txns.append((tx_dt, t))

    # Group by merchant (case-insensitive)
    by_merchant: dict[str, list[tuple]] = defaultdict(list)
    for dt, t in expense_txns:
        by_merchant[t.merchant.lower()].append((dt, t))

    flags: list[AnomalyFlag] = []
    window = timedelta(days=FREQUENCY_WINDOW_DAYS)
    flagged_ids: set[str] = set()

    for merchant, entries in by_merchant.items():
        entries_sorted = sorted(entries, key=lambda x: x[0])

        for i, (dt_i, t_i) in enumerate(entries_sorted):
            # Count how many transactions fall within [dt_i, dt_i + window] 日期窗口内的交易
            window_txns = [
                t for dt_j, t in entries_sorted[i:]
                if dt_j - dt_i <= window
            ]
            if len(window_txns) > FREQUENCY_THRESHOLD:
                # Flag excess transactions (those beyond the threshold)
                for excess_t in window_txns[FREQUENCY_THRESHOLD:]:
                    if excess_t.id not in flagged_ids:
                        flagged_ids.add(excess_t.id)
                        flags.append(
                            AnomalyFlag(
                                transaction_id=excess_t.id,
                                anomaly_type=AnomalyType.UNUSUAL_FREQUENCY,
                                explanation=(
                                    f"Merchant '{excess_t.merchant}' appeared "
                                    f"{len(window_txns)} times within "
                                    f"{FREQUENCY_WINDOW_DAYS} days "
                                    f"(threshold: {FREQUENCY_THRESHOLD})."
                                ),
                            )
                        )

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_anomalies(transactions: list[Transaction]) -> list[AnomalyFlag]:
    """
    Run all anomaly detectors against the given transaction list.

    Returns a list of AnomalyFlag objects sorted by (transaction_id, anomaly_type).
    LLM-based explanation is handled by extractor.py.
    """
    flags: list[AnomalyFlag] = []
    flags.extend(_detect_unusual_amounts(transactions))
    flags.extend(_detect_unusual_frequency(transactions))
    return sorted(flags, key=lambda f: (f.transaction_id, f.anomaly_type))
