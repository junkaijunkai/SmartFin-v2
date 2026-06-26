"""
Financial Health and Risk Assessment.

Two layers:

  Deterministic layer (always runs)
  ──────────────────────────────────
  1. debt_to_income_ratio  — housing + utilities spending as a fraction of income.
  2. liquid_reserve_months — months of average spending covered by savings balance.
  3. income_concentration_risk — True when ≥ CONCENTRATION_THRESHOLD of all income
                                  flows from a single merchant/source.
  4. sustained_overspending — True when total non-savings expense spending over the
                               most recent PERIOD_DAYS window exceeds monthly income.

  LLM advisory layer (best-effort, silent fallback)
  ──────────────────────────────────────────────────
  After the deterministic metrics and rating are computed, _generate_advisory()
  calls Claude to produce 2–4 plain-English personalised observations.  If the
  LLM is unavailable or fails after MAX_RETRIES attempts, _build_observations()
  is used as a silent fallback — callers always receive a populated observations
  list regardless of LLM availability.

Overall HealthRating thresholds
────────────────────────────────
  good  DTI < DTI_GOOD  AND  reserves ≥ RESERVE_GOOD  AND  no risk flags
  fair  DTI < DTI_POOR  AND  (reserves ≥ RESERVE_FAIR  OR  ≤ 1 risk flag)
  poor  anything worse
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from app.config import get_llm, get_prompt
from app.state import (
    Alert,
    AlertSeverity,
    HealthRating,
    HealthSummary,
    SpendingTrend,
    Transaction,
    TransactionCategory,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

PERIOD_DAYS = 30  # rolling window used for expense totals

# Debt-to-income thresholds (fraction of monthly income)
DTI_GOOD = 0.36
DTI_POOR = 0.50

# Liquid reserve thresholds (months of average spending)
RESERVE_GOOD = 3.0
RESERVE_FAIR = 1.0

# Income concentration: flag when one source represents ≥ this share of total income
CONCENTRATION_THRESHOLD = 0.90

# Categories that represent fixed obligations (used for DTI numerator)
_OBLIGATION_CATEGORIES = {TransactionCategory.HOUSING, TransactionCategory.UTILITIES}

# Category used to measure savings balance
_SAVINGS_CATEGORY = TransactionCategory.SAVINGS

# ---------------------------------------------------------------------------
# LLM advisory constants  (mirrors pattern in expense_analysis/categoriser.py)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal metric helpers
# ---------------------------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    """Return timezone-aware UTC datetime."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _compute_dti(
    transactions: list[Transaction],
    monthly_income: float,
    reference_date: datetime | None = None,
) -> float:
    """
    Debt-to-income ratio: sum of housing + utilities expenses in the last
    PERIOD_DAYS divided by monthly_income.

    Returns 0.0 when monthly_income is zero to avoid division by zero.
    """
    if monthly_income <= 0:
        return 0.0

    ref = reference_date or datetime.now(tz=timezone.utc)
    cutoff = ref - timedelta(days=PERIOD_DAYS)
    obligation_total = sum(
        t.amount
        for t in transactions
        if t.amount > 0
        and t.category in _OBLIGATION_CATEGORIES
        and _utc(t.date) >= cutoff
    )
    return round(obligation_total / monthly_income, 4)


def _compute_reserve_months(
    transactions: list[Transaction],
    monthly_income: float,
    reference_date: datetime | None = None,
) -> float:
    """
    Liquid reserve months: cumulative savings deposits divided by average
    monthly expense spending.

    Savings deposits are identified by TransactionCategory.SAVINGS with
    amount > 0 (positive = money moved into savings, per the sign convention).

    Falls back to monthly_income as the denominator when no expense
    transactions are present (avoids division by zero).
    """
    ref = reference_date or datetime.now(tz=timezone.utc)
    cutoff = ref - timedelta(days=PERIOD_DAYS)

    savings_balance = sum(
        t.amount
        for t in transactions
        if t.amount > 0 and t.category == _SAVINGS_CATEGORY
    )

    monthly_expenses = sum(
        t.amount
        for t in transactions
        if t.amount > 0
        and t.category != _SAVINGS_CATEGORY
        and _utc(t.date) >= cutoff
    )

    denominator = monthly_expenses if monthly_expenses > 0 else max(monthly_income, 1.0)
    return round(savings_balance / denominator, 4)


def _detect_income_concentration(transactions: list[Transaction]) -> bool:
    """
    Returns True when a single merchant/source accounts for ≥
    CONCENTRATION_THRESHOLD of total income (amount < 0 = income).
    """
    income_by_merchant: dict[str, float] = defaultdict(float)
    for t in transactions:
        if t.amount < 0:
            income_by_merchant[t.merchant.lower()] += abs(t.amount)

    total_income = sum(income_by_merchant.values())
    if total_income == 0:
        return False

    max_share = max(income_by_merchant.values()) / total_income
    return max_share >= CONCENTRATION_THRESHOLD


def _detect_sustained_overspending(
    transactions: list[Transaction],
    monthly_income: float,
    reference_date: datetime | None = None,
) -> bool:
    """
    Returns True when total non-savings expense spending over the last
    PERIOD_DAYS exceeds monthly_income.

    Savings deposits are excluded: moving money into savings is prudent
    behaviour, not overspending.
    """
    if monthly_income <= 0:
        return False

    ref = reference_date or datetime.now(tz=timezone.utc)
    cutoff = ref - timedelta(days=PERIOD_DAYS)
    total_expenses = sum(
        t.amount
        for t in transactions
        if t.amount > 0
        and t.category != _SAVINGS_CATEGORY
        and _utc(t.date) >= cutoff
    )
    return total_expenses > monthly_income


# ---------------------------------------------------------------------------
# Rating logic
# ---------------------------------------------------------------------------


def _derive_rating(
    dti: float,
    reserve_months: float,
    income_concentration_risk: bool,
    sustained_overspending: bool,
) -> HealthRating:
    """
    Map computed metrics to a HealthRating.

    good  → DTI < DTI_GOOD  AND reserves ≥ RESERVE_GOOD  AND no risk flags
    poor  → DTI ≥ DTI_POOR  OR  reserves < RESERVE_FAIR  OR  both risk flags set
    fair  → everything in between
    """
    risk_flags = int(income_concentration_risk) + int(sustained_overspending)

    if dti < DTI_GOOD and reserve_months >= RESERVE_GOOD and risk_flags == 0:
        return HealthRating.GOOD

    if dti >= DTI_POOR or reserve_months < RESERVE_FAIR or risk_flags >= 2:
        return HealthRating.POOR

    return HealthRating.FAIR


def _build_observations(
    dti: float,
    reserve_months: float,
    income_concentration_risk: bool,
    sustained_overspending: bool,
) -> list[str]:
    """Return human-readable observation strings for the HealthSummary."""
    obs: list[str] = []

    if dti < DTI_GOOD:
        obs.append(
            f"Debt-to-income ratio is {dti:.0%} — within the healthy range (< {DTI_GOOD:.0%})."
        )
    elif dti < DTI_POOR:
        obs.append(
            f"Debt-to-income ratio is {dti:.0%} — elevated (target: < {DTI_GOOD:.0%})."
        )
    else:
        obs.append(
            f"Debt-to-income ratio is {dti:.0%} — high risk (threshold: {DTI_POOR:.0%})."
        )

    if reserve_months >= RESERVE_GOOD:
        obs.append(
            f"Liquid reserves cover {reserve_months:.1f} months of spending — adequate."
        )
    elif reserve_months >= RESERVE_FAIR:
        obs.append(
            f"Liquid reserves cover {reserve_months:.1f} months of spending — consider building to {RESERVE_GOOD:.0f} months."
        )
    else:
        obs.append(
            f"Liquid reserves cover only {reserve_months:.1f} months of spending — critically low."
        )

    if income_concentration_risk:
        obs.append(
            f"Income concentration risk detected: ≥ {CONCENTRATION_THRESHOLD:.0%} of income "
            "comes from a single source."
        )

    if sustained_overspending:
        obs.append(
            "Sustained overspending detected: total expenses over the last "
            f"{PERIOD_DAYS} days exceed monthly income."
        )

    return obs


# ---------------------------------------------------------------------------
# LLM advisory layer
# ---------------------------------------------------------------------------


class _AdvisoryResult(BaseModel):
    observations: list[str] = Field(
        description=(
            "2–4 plain-English advisory observations about the user's financial health, "
            "each 1–2 sentences long"
        )
    )




def _generate_advisory(
    rating: HealthRating,
    dti: float,
    reserve_months: float,
    concentration_risk: bool,
    overspending: bool,
    monthly_income: float,
    spending_trends: list[SpendingTrend],
) -> list[str] | None:
    """
    Call Claude to generate personalised advisory observations.

    Follows the same retry + fallback pattern as expense_analysis/categoriser.py.

    Returns a list of plain-English observation strings on success, or None if
    the LLM is unavailable — the caller falls back to _build_observations().
    """
    try:
        llm = get_llm("default", timeout=_LLM_TIMEOUT, temperature=0.7)
        structured_llm = llm.with_structured_output(_AdvisoryResult)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for health advisory: %s", exc)
        return None

    trend_lines: list[str] = []
    for t in sorted(spending_trends, key=lambda s: s.current_period_total, reverse=True)[:3]:
        if t.deviation_pct is not None:
            sign = "+" if t.deviation_pct >= 0 else ""
            trend_lines.append(
                f"  {t.category.value}: £{t.current_period_total:.0f}/month "
                f"({sign}{t.deviation_pct:.1f}% vs prior period)"
            )
        else:
            trend_lines.append(
                f"  {t.category.value}: £{t.current_period_total:.0f}/month (no prior period data)"
            )
    trends_text = "\n".join(trend_lines) if trend_lines else "  No spending trend data available."

    messages = get_prompt("health_advisory").format_messages(
        rating=rating.value.upper(),
        dti=f"{dti:.0%}",
        reserve_months=f"{reserve_months:.1f}",
        concentration_risk="Yes — single income source dominates" if concentration_risk else "No",
        overspending="Yes — expenses exceed monthly income" if overspending else "No",
        monthly_income=f"{monthly_income:.0f}",
        trends_text=trends_text,
    )
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response: _AdvisoryResult = structured_llm.invoke(messages)
            return response.observations
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "LLM advisory generation failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM advisory generation failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_health(
    transactions: list[Transaction],
    monthly_income: float,
    spending_trends: list[SpendingTrend] | None = None,
    reference_date: datetime | None = None,
) -> tuple[HealthSummary, list[Alert]]:
    """
    Compute a HealthSummary and any associated Alerts from the given data.

    Parameters
    ----------
    transactions:    categorised (or raw) transaction list
    monthly_income:  user-declared monthly income (positive float)
    spending_trends: pre-computed trends from expense_analysis; forwarded to the
                     LLM advisory layer so it can reference accelerating categories
    reference_date:  the "current date" for computing rolling 30-day windows.
                     Defaults to ``datetime.now(tz=timezone.utc)``.

    Returns
    -------
    (HealthSummary, list[Alert])
        observations is populated by the LLM when available, falling back silently
        to _build_observations() rule strings if the LLM is unavailable.
        Alerts are generated only for poor/fair ratings with actionable findings.
    """
    import uuid

    ref = reference_date or datetime.now(tz=timezone.utc)
    dti = _compute_dti(transactions, monthly_income, ref)
    reserve_months = _compute_reserve_months(transactions, monthly_income, ref)
    concentration_risk = _detect_income_concentration(transactions)
    overspending = _detect_sustained_overspending(transactions, monthly_income, ref)

    rating = _derive_rating(dti, reserve_months, concentration_risk, overspending)

    llm_observations = _generate_advisory(
        rating=rating,
        dti=dti,
        reserve_months=reserve_months,
        concentration_risk=concentration_risk,
        overspending=overspending,
        monthly_income=monthly_income,
        spending_trends=spending_trends or [],
    )
    observations = llm_observations if llm_observations is not None else _build_observations(
        dti, reserve_months, concentration_risk, overspending
    )

    summary = HealthSummary(
        rating=rating,
        debt_to_income_ratio=dti,
        liquid_reserve_months=reserve_months,
        income_concentration_risk=concentration_risk,
        sustained_overspending=overspending,
        observations=observations,
    )

    alerts: list[Alert] = []

    if rating == HealthRating.POOR:
        alerts.append(
            Alert(
                id=str(uuid.uuid4()),
                severity=AlertSeverity.CRITICAL,
                source_agent="health_assessment",
                message=(
                    f"Financial health is POOR. "
                    f"DTI: {dti:.0%}, reserves: {reserve_months:.1f} months. "
                    "Immediate review recommended."
                ),
            )
        )
    elif rating == HealthRating.FAIR:
        alerts.append(
            Alert(
                id=str(uuid.uuid4()),
                severity=AlertSeverity.WARNING,
                source_agent="health_assessment",
                message=(
                    f"Financial health is FAIR. "
                    f"DTI: {dti:.0%}, reserves: {reserve_months:.1f} months. "
                    "Review spending in flagged areas."
                ),
            )
        )

    return summary, alerts
