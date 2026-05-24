"""
Spending trend analyser — pure Python, no LLM.

Compares the last 30 days (current period) against the 30 days before that
(previous period) and returns a SpendingTrend per category.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from app.state import SpendingTrend, Transaction, TransactionCategory

PERIOD_DAYS = 30  # length of each comparison window in days


def compute_spending_trends(transactions: list[Transaction]) -> list[SpendingTrend]:
    """
    Compute per-category spending trends from a list of categorised transactions.

    Only expenses (amount > 0) contribute to the totals.
    Categories with no activity in either period are excluded.
    Results are sorted by current-period spend, descending.
    """

    now = datetime.now(tz=timezone.utc)
    current_start = now - timedelta(days=PERIOD_DAYS)
    previous_start = now - timedelta(days=PERIOD_DAYS * 2)

    current_totals: dict[TransactionCategory, float] = defaultdict(float)
    previous_totals: dict[TransactionCategory, float] = defaultdict(float)

    for t in transactions:
        if t.amount <= 0:
            continue  # skip income entries

        # Normalise to UTC-aware datetime
        tx_dt = t.date
        if tx_dt.tzinfo is None:
            tx_dt = tx_dt.replace(tzinfo=timezone.utc)

        if tx_dt >= current_start:
            current_totals[t.category] += t.amount
        elif tx_dt >= previous_start:
            previous_totals[t.category] += t.amount

    all_categories = set(current_totals) | set(previous_totals)

    trends: list[SpendingTrend] = []
    for category in all_categories:
        current = current_totals.get(category, 0.0)
        previous = previous_totals.get(category, 0.0)

        if previous > 0 and current > 0:
            deviation_pct = round((current - previous) / previous * 100, 2)
        else:
            deviation_pct = None

        trends.append(
            SpendingTrend(
                category=category,
                current_period_total=round(current, 2),
                previous_period_total=round(previous, 2),
                deviation_pct=deviation_pct,
            )
        )

    return sorted(trends, key=lambda s: s.current_period_total, reverse=True)
