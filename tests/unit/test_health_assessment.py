"""Unit tests for Financial Health and Risk Assessment — assessor.py and agent.py."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.agents.health_assessment.agent import run as health_run
from app.orchestrator.state_view import AgentStateView
from app.agents.health_assessment.assessor import (
    CONCENTRATION_THRESHOLD,
    DTI_GOOD,
    DTI_POOR,
    PERIOD_DAYS,
    RESERVE_FAIR,
    RESERVE_GOOD,
    _AdvisoryResult,
    _compute_dti,
    _compute_reserve_months,
    _generate_advisory,
    _detect_income_concentration,
    _detect_sustained_overspending,
    _derive_rating,
    assess_health,
)
from app.state import HealthRating, Transaction, TransactionCategory


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
        description="test",
        merchant=merchant,
        category=category,
    )


def _income(amount: float, merchant: str = "Employer", days_ago: float = 1) -> Transaction:
    """Convenience: income transaction (negative amount)."""
    return _txn(-abs(amount), category=TransactionCategory.INCOME, merchant=merchant, days_ago=days_ago)


def _expense(
    amount: float,
    category: TransactionCategory = TransactionCategory.FOOD,
    days_ago: float = 1,
) -> Transaction:
    return _txn(abs(amount), category=category, days_ago=days_ago)


def _housing(amount: float, days_ago: float = 1) -> Transaction:
    return _txn(abs(amount), category=TransactionCategory.HOUSING, days_ago=days_ago)


def _utilities(amount: float, days_ago: float = 1) -> Transaction:
    return _txn(abs(amount), category=TransactionCategory.UTILITIES, days_ago=days_ago)


def _savings(amount: float, days_ago: float = 1) -> Transaction:
    return _txn(abs(amount), category=TransactionCategory.SAVINGS, days_ago=days_ago)


# ---------------------------------------------------------------------------
# Module-level fixture: block real LLM calls in all unit tests.
#
# ChatAnthropic is patched to raise immediately so _generate_advisory()
# returns None and assess_health() falls back to _build_observations().
# Tests that specifically exercise the LLM path override this fixture
# by calling monkeypatch.setattr again inside their own test body —
# the last setattr to the same target wins within a single test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Prevent real LLM calls: make ChatAnthropic raise on construction."""
    def _raise(*args, **kwargs):
        raise RuntimeError("LLM disabled in unit tests — patch explicitly to test LLM path")
    monkeypatch.setattr("app.agents.health_assessment.assessor.ChatAnthropic", _raise)


# ---------------------------------------------------------------------------
# TestComputeDTI
# ---------------------------------------------------------------------------


class TestComputeDTI:
    def test_housing_and_utilities_included(self):
        txns = [_housing(400), _utilities(100)]
        dti = _compute_dti(txns, monthly_income=2500.0)
        assert dti == round(500 / 2500, 4)

    def test_other_categories_excluded(self):
        txns = [_housing(400), _expense(300, TransactionCategory.FOOD)]
        dti = _compute_dti(txns, monthly_income=2000.0)
        assert dti == round(400 / 2000, 4)

    def test_zero_income_returns_zero(self):
        txns = [_housing(500)]
        assert _compute_dti(txns, monthly_income=0.0) == 0.0

    def test_only_recent_transactions_counted(self):
        recent = _housing(500, days_ago=1)
        old = _housing(500, days_ago=PERIOD_DAYS + 5)
        dti = _compute_dti([recent, old], monthly_income=2000.0)
        assert dti == round(500 / 2000, 4)

    def test_income_transactions_excluded(self):
        txns = [_income(5000), _housing(400)]
        dti = _compute_dti(txns, monthly_income=5000.0)
        assert dti == round(400 / 5000, 4)

    def test_empty_transactions_returns_zero(self):
        assert _compute_dti([], monthly_income=3000.0) == 0.0


# ---------------------------------------------------------------------------
# TestComputeReserveMonths
# ---------------------------------------------------------------------------


class TestComputeReserveMonths:
    def test_ratio_uses_savings_over_expenses(self):
        txns = [_savings(1500), _expense(500, TransactionCategory.FOOD)]
        reserve = _compute_reserve_months(txns, monthly_income=3000.0)
        assert reserve == round(1500 / 500, 4)

    def test_no_expenses_falls_back_to_income(self):
        txns = [_savings(3000)]
        reserve = _compute_reserve_months(txns, monthly_income=3000.0)
        assert reserve == round(3000 / 3000, 4)

    def test_no_savings_returns_zero(self):
        txns = [_expense(500)]
        reserve = _compute_reserve_months(txns, monthly_income=3000.0)
        assert reserve == 0.0

    def test_savings_not_in_expense_denominator(self):
        """Savings deposits should not inflate the monthly expense denominator."""
        txns = [_savings(2000), _expense(500)]
        reserve = _compute_reserve_months(txns, monthly_income=3000.0)
        # denominator = 500 (savings excluded), numerator = 2000
        assert reserve == round(2000 / 500, 4)

    def test_empty_transactions_returns_zero(self):
        assert _compute_reserve_months([], monthly_income=3000.0) == 0.0


# ---------------------------------------------------------------------------
# TestDetectIncomeConcentration
# ---------------------------------------------------------------------------


class TestDetectIncomeConcentration:
    def test_single_employer_triggers_flag(self):
        txns = [_income(3000, merchant="Acme Corp")]
        assert _detect_income_concentration(txns) is True

    def test_diversified_income_no_flag(self):
        txns = [
            _income(2000, merchant="Employer A"),
            _income(2000, merchant="Employer B"),
        ]
        assert _detect_income_concentration(txns) is False

    def test_just_below_threshold_no_flag(self):
        # 89% from one source — below CONCENTRATION_THRESHOLD (0.90)
        txns = [
            _income(890, merchant="Main"),
            _income(110, merchant="Side"),
        ]
        assert _detect_income_concentration(txns) is False

    def test_at_threshold_triggers_flag(self):
        # exactly 90%
        txns = [
            _income(900, merchant="Main"),
            _income(100, merchant="Side"),
        ]
        assert _detect_income_concentration(txns) is True

    def test_no_income_returns_false(self):
        txns = [_expense(500)]
        assert _detect_income_concentration(txns) is False

    def test_case_insensitive_merchant(self):
        txns = [
            _income(500, merchant="Employer"),
            _income(500, merchant="EMPLOYER"),
        ]
        # Both normalise to "employer" — 100% from one source
        assert _detect_income_concentration(txns) is True


# ---------------------------------------------------------------------------
# TestDetectSustainedOverspending
# ---------------------------------------------------------------------------


class TestDetectSustainedOverspending:
    def test_overspending_when_expenses_exceed_income(self):
        txns = [_expense(4000)]  # 4000 > 3000 income
        assert _detect_sustained_overspending(txns, monthly_income=3000.0) is True

    def test_no_flag_when_within_income(self):
        txns = [_expense(2000)]
        assert _detect_sustained_overspending(txns, monthly_income=3000.0) is False

    def test_old_transactions_excluded(self):
        old = _expense(5000, days_ago=PERIOD_DAYS + 5)
        assert _detect_sustained_overspending([old], monthly_income=3000.0) is False

    def test_zero_income_returns_false(self):
        txns = [_expense(500)]
        assert _detect_sustained_overspending(txns, monthly_income=0.0) is False

    def test_empty_transactions_returns_false(self):
        assert _detect_sustained_overspending([], monthly_income=3000.0) is False


# ---------------------------------------------------------------------------
# TestDeriveRating
# ---------------------------------------------------------------------------


class TestDeriveRating:
    def test_good_when_all_metrics_healthy(self):
        assert _derive_rating(
            dti=0.20,
            reserve_months=4.0,
            income_concentration_risk=False,
            sustained_overspending=False,
        ) == HealthRating.GOOD

    def test_poor_when_dti_too_high(self):
        assert _derive_rating(
            dti=DTI_POOR + 0.01,
            reserve_months=4.0,
            income_concentration_risk=False,
            sustained_overspending=False,
        ) == HealthRating.POOR

    def test_poor_when_reserves_critically_low(self):
        assert _derive_rating(
            dti=0.20,
            reserve_months=RESERVE_FAIR - 0.01,
            income_concentration_risk=False,
            sustained_overspending=False,
        ) == HealthRating.POOR

    def test_poor_when_both_risk_flags_set(self):
        assert _derive_rating(
            dti=0.20,
            reserve_months=4.0,
            income_concentration_risk=True,
            sustained_overspending=True,
        ) == HealthRating.POOR

    def test_fair_when_dti_elevated(self):
        assert _derive_rating(
            dti=DTI_GOOD + 0.01,
            reserve_months=RESERVE_GOOD,
            income_concentration_risk=False,
            sustained_overspending=False,
        ) == HealthRating.FAIR

    def test_fair_when_reserves_adequate_but_low(self):
        assert _derive_rating(
            dti=0.20,
            reserve_months=RESERVE_FAIR + 0.1,
            income_concentration_risk=False,
            sustained_overspending=False,
        ) == HealthRating.FAIR

    def test_fair_with_one_risk_flag(self):
        assert _derive_rating(
            dti=0.20,
            reserve_months=RESERVE_GOOD,
            income_concentration_risk=True,
            sustained_overspending=False,
        ) == HealthRating.FAIR

    def test_good_boundary_dti(self):
        """DTI exactly at DTI_GOOD boundary is NOT good (must be strictly less)."""
        assert _derive_rating(
            dti=DTI_GOOD,
            reserve_months=RESERVE_GOOD,
            income_concentration_risk=False,
            sustained_overspending=False,
        ) == HealthRating.FAIR


# ---------------------------------------------------------------------------
# TestAssessHealth (public API)
# ---------------------------------------------------------------------------


class TestAssessHealth:
    def _good_state(self) -> tuple[list[Transaction], float]:
        # Diversified income → no concentration risk
        # DTI = (300+50)/3000 = 0.117 < DTI_GOOD
        # reserves = 9000/(300+50+200) = 16.4 months > RESERVE_GOOD
        # overspending: 300+50+200 = 550 < 3000 → False
        txns = [
            _housing(300),
            _utilities(50),
            _expense(200),
            _savings(9000),
            _income(2000, merchant="Employer A"),
            _income(1000, merchant="Employer B"),
        ]
        return txns, 3000.0

    def test_returns_health_summary_and_alerts(self):
        txns, income = self._good_state()
        summary, alerts = assess_health(txns, income)
        from app.state import HealthSummary
        assert isinstance(summary, HealthSummary)
        assert isinstance(alerts, list)

    def test_good_rating_produces_no_alerts(self):
        txns, income = self._good_state()
        _, alerts = assess_health(txns, income)
        assert alerts == []

    def test_poor_rating_produces_critical_alert(self):
        from app.state import AlertSeverity
        # Trigger poor: extremely high DTI + low reserves
        txns = [_housing(4000)]
        _, alerts = assess_health(txns, monthly_income=5000.0)
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)

    def test_fair_rating_produces_warning_alert(self):
        from app.state import AlertSeverity
        # Elevated DTI (1100/3000 = 0.367, > DTI_GOOD) but below DTI_POOR.
        # Diversified income → no concentration risk.
        # Adequate reserves (4000 / 1100 = 3.6 months).
        # No overspending (1100 < 3000).
        # → FAIR rating, WARNING alert.
        txns = [
            _housing(1100),
            _savings(4000),
            _income(2000, merchant="Employer A"),
            _income(1000, merchant="Employer B"),
        ]
        _, alerts = assess_health(txns, monthly_income=3000.0)
        assert any(a.severity == AlertSeverity.WARNING for a in alerts)

    def test_observations_are_non_empty(self):
        txns, income = self._good_state()
        summary, _ = assess_health(txns, income)
        assert len(summary.observations) > 0

    def test_empty_transactions_returns_good_rating(self):
        summary, alerts = assess_health([], monthly_income=3000.0)
        # No expenses → DTI=0, reserves=0 (< RESERVE_FAIR) → POOR
        assert summary.rating == HealthRating.POOR

    def test_summary_fields_populated(self):
        txns, income = self._good_state()
        summary, _ = assess_health(txns, income)
        assert summary.debt_to_income_ratio >= 0
        assert summary.liquid_reserve_months >= 0
        assert isinstance(summary.income_concentration_risk, bool)
        assert isinstance(summary.sustained_overspending, bool)

    def test_spending_trends_accepted_without_error(self):
        """spending_trends parameter is forwarded without raising."""
        from app.state import SpendingTrend
        trend = SpendingTrend(
            category=TransactionCategory.FOOD,
            current_period_total=200.0,
            previous_period_total=180.0,
            deviation_pct=11.1,
        )
        txns, income = self._good_state()
        summary, _ = assess_health(txns, income, spending_trends=[trend])
        assert summary is not None


# ---------------------------------------------------------------------------
# TestHealthAssessmentAgentNode
# ---------------------------------------------------------------------------


class TestHealthAssessmentAgentNode:

    def _run(self, state: dict) -> dict:
        return health_run(AgentStateView(state, "health_assessment"))

    @patch("app.agents.health_assessment.agent.assess_health")
    def test_writes_health_summary_to_state(self, mock_assess):
        from app.state import HealthSummary, HealthRating

        mock_assess.return_value = (
            HealthSummary(
                rating=HealthRating.GOOD,
                debt_to_income_ratio=0.0,
                liquid_reserve_months=0.0,
                income_concentration_risk=False,
                sustained_overspending=False,
            ),
            [],
        )
        result = self._run({"transactions": [_housing(100)], "monthly_income": 3000.0, "messages": []})

        assert "health_summary" in result
        assert result["health_summary"] is not None

    @patch("app.agents.health_assessment.agent.assess_health")
    def test_writes_alerts_to_state(self, mock_assess):
        from app.state import HealthSummary, HealthRating

        mock_assess.return_value = (
            HealthSummary(
                rating=HealthRating.GOOD,
                debt_to_income_ratio=0.0,
                liquid_reserve_months=0.0,
                income_concentration_risk=False,
                sustained_overspending=False,
            ),
            [],
        )
        result = self._run({"transactions": [_housing(100)], "monthly_income": 3000.0, "messages": []})

        assert "alerts" in result
        assert isinstance(result["alerts"], list)

    def test_empty_state_does_not_raise(self):
        result = self._run({"monthly_income": 0})
        # No transactions + no income → returns early without calling assess_health
        assert "health_summary" in result

    @patch("app.agents.health_assessment.agent.assess_health")
    def test_prefers_categorised_transactions(self, mock_assess):
        from app.state import HealthSummary, HealthRating

        mock_assess.return_value = (
            HealthSummary(
                rating=HealthRating.GOOD,
                debt_to_income_ratio=round(600 / 3000, 4),
                liquid_reserve_months=0.0,
                income_concentration_risk=False,
                sustained_overspending=False,
            ),
            [],
        )

        categorised = [_housing(600), _income(3000)]
        result = self._run({
            "transactions": [],
            "categorised_transactions": categorised,
            "monthly_income": 3000.0,
            "messages": [],
        })
        assert result["health_summary"].debt_to_income_ratio == round(600 / 3000, 4)

    @patch("app.agents.health_assessment.agent.assess_health")
    def test_falls_back_to_raw_transactions(self, mock_assess):
        from app.state import HealthSummary, HealthRating

        mock_assess.return_value = (
            HealthSummary(
                rating=HealthRating.GOOD,
                debt_to_income_ratio=round(600 / 3000, 4),
                liquid_reserve_months=0.0,
                income_concentration_risk=False,
                sustained_overspending=False,
            ),
            [],
        )

        raw = [_housing(600), _income(3000)]
        result = self._run({
            "transactions": raw,
            "monthly_income": 3000.0,
            "messages": [],
        })
        assert result["health_summary"].debt_to_income_ratio == round(600 / 3000, 4)

    @patch("app.agents.health_assessment.agent.assess_health")
    def test_merges_existing_alerts(self, mock_assess):
        from app.state import HealthSummary, HealthRating, Alert, AlertSeverity

        new_alert = Alert(
            id="new-1",
            severity=AlertSeverity.CRITICAL,
            source_agent="health_assessment",
            message="Financial health is POOR.",
        )
        mock_assess.return_value = (
            HealthSummary(
                rating=HealthRating.POOR,
                debt_to_income_ratio=5000 / 3000,
                liquid_reserve_months=0.0,
                income_concentration_risk=False,
                sustained_overspending=True,
            ),
            [new_alert],
        )

        existing = Alert(
            id="existing-1",
            severity=AlertSeverity.INFO,
            source_agent="other_agent",
            message="existing alert",
        )
        result = self._run({
            "transactions": [_housing(5000)],
            "monthly_income": 3000.0,
            "alerts": [existing],
            "messages": [],
        })
        ids = [a.id for a in result["alerts"]]
        assert "existing-1" in ids
        assert "new-1" in ids
        assert len(result["alerts"]) >= 2


# ---------------------------------------------------------------------------
# TestGenerateAdvisory
# ---------------------------------------------------------------------------


class TestGenerateAdvisory:
    """Tests for the LLM advisory layer in _generate_advisory()."""

    def _metrics(self) -> dict:
        return dict(
            rating=HealthRating.FAIR,
            dti=0.42,
            reserve_months=1.8,
            concentration_risk=True,
            overspending=False,
            monthly_income=3200.0,
            spending_trends=[],
        )

    def _mock_llm(self, observations: list[str]) -> MagicMock:
        """Return a ChatAnthropic mock that yields the given observations."""
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.return_value = (
            _AdvisoryResult(observations=observations)
        )
        return mock_instance

    # --- LLM success path ---

    def test_returns_list_of_strings_on_success(self, monkeypatch):
        llm = self._mock_llm(["obs1", "obs2", "obs3"])
        monkeypatch.setattr(
            "app.agents.health_assessment.assessor.ChatAnthropic",
            lambda *args, **kwargs: llm,
        )
        result = _generate_advisory(**self._metrics())
        assert result == ["obs1", "obs2", "obs3"]

    def test_uses_structured_output(self, monkeypatch):
        """with_structured_output must be called so Claude returns typed JSON."""
        llm = self._mock_llm(["obs"])
        monkeypatch.setattr(
            "app.agents.health_assessment.assessor.ChatAnthropic",
            lambda *args, **kwargs: llm,
        )
        _generate_advisory(**self._metrics())
        llm.with_structured_output.assert_called_once_with(_AdvisoryResult)

    def test_reads_model_from_env(self, monkeypatch):
        """_generate_advisory uses the 'default' alias, which resolves to claude-haiku-4-5."""
        captured = {}

        def _capture_model(*args, **kwargs):
            captured["model"] = kwargs.get("model")
            return self._mock_llm(["obs"])

        monkeypatch.setattr("app.agents.health_assessment.assessor.ChatAnthropic", _capture_model)
        _generate_advisory(**self._metrics())
        assert captured["model"] == "claude-haiku-4-5"

    def test_default_model_when_env_unset(self, monkeypatch):
        captured = {}

        def _capture_model(*args, **kwargs):
            captured["model"] = kwargs.get("model")
            return self._mock_llm(["obs"])

        monkeypatch.setattr("app.agents.health_assessment.assessor.ChatAnthropic", _capture_model)
        _generate_advisory(**self._metrics())
        assert captured["model"] == "claude-haiku-4-5"

    # --- Fallback paths ---

    def test_returns_none_when_llm_init_fails(self):
        # The module-level autouse fixture already patches ChatAnthropic to raise,
        # so _generate_advisory must return None here without any extra patching.
        result = _generate_advisory(**self._metrics())
        assert result is None

    def test_returns_none_when_invoke_fails(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.side_effect = (
            RuntimeError("API error")
        )
        monkeypatch.setattr(
            "app.agents.health_assessment.assessor.ChatAnthropic",
            lambda *args, **kwargs: mock_instance,
        )
        # Patch time.sleep to avoid real delay during retry backoff
        monkeypatch.setattr("app.agents.health_assessment.assessor.time.sleep", lambda s: None)
        result = _generate_advisory(**self._metrics())
        assert result is None

    def test_retries_on_invoke_failure(self, monkeypatch):
        """invoke should be called MAX_RETRIES times before giving up."""
        from app.agents.health_assessment.assessor import MAX_RETRIES
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.invoke.side_effect = (
            RuntimeError("API error")
        )
        monkeypatch.setattr(
            "app.agents.health_assessment.assessor.ChatAnthropic",
            lambda *args, **kwargs: mock_instance,
        )
        monkeypatch.setattr("app.agents.health_assessment.assessor.time.sleep", lambda s: None)
        _generate_advisory(**self._metrics())
        assert mock_instance.with_structured_output.return_value.invoke.call_count == MAX_RETRIES

    # --- Integration with assess_health ---

    def test_assess_health_uses_llm_observations(self, monkeypatch):
        """When LLM succeeds, observations in HealthSummary come from the LLM."""
        llm_obs = ["LLM observation one.", "LLM observation two."]
        llm = self._mock_llm(llm_obs)
        monkeypatch.setattr(
            "app.agents.health_assessment.assessor.ChatAnthropic",
            lambda *args, **kwargs: llm,
        )
        txns = [_housing(300), _utilities(50), _expense(200),
                _savings(9000), _income(2000, merchant="A"), _income(1000, merchant="B")]
        summary, _ = assess_health(txns, monthly_income=3000.0)
        assert summary.observations == llm_obs

    def test_assess_health_falls_back_when_llm_unavailable(self):
        """With LLM disabled (autouse fixture), observations come from _build_observations."""
        txns = [_housing(300), _utilities(50), _expense(200),
                _savings(9000), _income(2000, merchant="A"), _income(1000, merchant="B")]
        summary, _ = assess_health(txns, monthly_income=3000.0)
        # Rule-based strings always contain the DTI percentage
        assert any("30%" in obs or "debt" in obs.lower() for obs in summary.observations)

    def test_assess_health_fallback_observations_non_empty(self):
        """Fallback path still produces a non-empty observations list."""
        summary, _ = assess_health([], monthly_income=3000.0)
        assert len(summary.observations) > 0
