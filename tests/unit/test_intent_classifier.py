"""
Unit tests for intent classification.

Tests both the keyword fallback and the LLM-based classification.
The LLM is mocked to ensure tests run without API keys.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.orchestrator.intent_classifier import (
    _IntentResult,
    _keyword_fallback,
    classify_intent,
)


# ---------------------------------------------------------------------------
# Tests for _keyword_fallback
# ---------------------------------------------------------------------------


class TestKeywordFallback:
    """Test keyword-based fallback routing."""

    def test_routes_to_budget_planning(self):
        """'budget' keyword routes to budget_planning."""
        assert _keyword_fallback("Help me with my budget") == "budget_planning"
        assert _keyword_fallback("BUDGET") == "budget_planning"

    def test_routes_to_goal_planning(self):
        """Goal-related keywords route to goal_planning."""
        assert _keyword_fallback("I want to save for a goal") == "goal_planning"
        assert _keyword_fallback("set a saving goal") == "goal_planning"
        assert _keyword_fallback("deposit money") == "goal_planning"
        assert _keyword_fallback("create a fund") == "goal_planning"

    def test_routes_to_anomaly_detection(self):
        """Anomaly keywords route to anomaly_detection."""
        assert _keyword_fallback("Find suspicious transactions") == "anomaly_detection"
        assert _keyword_fallback("anomaly") == "anomaly_detection"

    def test_routes_to_health_assessment(self):
        """Health keywords route to health_assessment."""
        assert _keyword_fallback("What is my financial health?") == "health_assessment"
        assert _keyword_fallback("risk assessment") == "health_assessment"

    def test_routes_to_expense_analysis(self):
        """Spending keywords route to expense_analysis."""
        assert _keyword_fallback("Show me my spending") == "expense_analysis"
        assert _keyword_fallback("transaction breakdown") == "expense_analysis"
        assert _keyword_fallback("expense report") == "expense_analysis"
        assert _keyword_fallback("categorize my spending") == "expense_analysis"

    def test_default_is_unknown(self):
        """Unmatched messages return unknown — out-of-scope messages should not be routed."""
        assert _keyword_fallback("Hello") == "unknown"
        assert _keyword_fallback("How are you?") == "unknown"
        assert _keyword_fallback("") == "unknown"


# ---------------------------------------------------------------------------
# Tests for classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntent:
    """Test LLM-based intent classification."""

    @patch("app.orchestrator.intent_classifier.get_cached_llm_response", return_value=None)
    @patch("app.orchestrator.intent_classifier.ChatAnthropic")
    def test_classify_intent_returns_llm_result(self, mock_llm_class, mock_cache):
        """classify_intent calls LLM and returns the agent name."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        result = _IntentResult(agent="budget_planning", reasoning="User mentioned budget")
        mock_structured.invoke.return_value = result

        # Call function
        agent = classify_intent("Help me with my budget")

        # Verify
        assert agent == "budget_planning"
        mock_llm_class.assert_called_once()
        mock_structured.invoke.assert_called_once()

    @patch("app.orchestrator.intent_classifier.get_cached_llm_response", return_value=None)
    @patch("app.orchestrator.intent_classifier.ChatAnthropic")
    def test_classify_intent_uses_correct_model(self, mock_llm_class, mock_cache, monkeypatch):
        """classify_intent uses SMARTFIN_MODEL env var, defaults to haiku."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = _IntentResult(
            agent="expense_analysis", reasoning="test"
        )

        # Test with env var set
        monkeypatch.setenv("SMARTFIN_MODEL", "claude-opus-4-7")
        classify_intent("test message")
        mock_llm_class.assert_called_with(model="claude-opus-4-7")

        # Reset and test default
        mock_llm_class.reset_mock()
        monkeypatch.delenv("SMARTFIN_MODEL")
        classify_intent("test message")
        mock_llm_class.assert_called_with(model="claude-haiku-4-5")

    @patch("app.orchestrator.intent_classifier.ChatAnthropic")
    def test_classify_intent_falls_back_to_keyword_on_exception(self, mock_llm_class):
        """classify_intent falls back to keyword matching if LLM raises exception."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        # Make invoke raise an exception
        mock_structured.invoke.side_effect = RuntimeError("API error")

        # Call with message that matches keyword
        agent = classify_intent("Show me my spending")

        # Should fall back to keyword matching
        assert agent == "expense_analysis"

    @patch("app.orchestrator.intent_classifier.ChatAnthropic")
    def test_classify_intent_fallback_preserves_keyword_logic(self, mock_llm_class):
        """Fallback respects keyword routing priorities."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.side_effect = RuntimeError("API error")

        # Test each keyword path
        assert classify_intent("Create a goal") == "goal_planning"
        assert classify_intent("Find anomalies") == "anomaly_detection"
        assert classify_intent("Check health") == "health_assessment"
        assert classify_intent("Plan budget") == "budget_planning"

    @patch("app.orchestrator.intent_classifier.ChatAnthropic")
    def test_classify_intent_routes_all_valid_agents(self, mock_llm_class):
        """classify_intent can route to all five agents."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        agents = [
            "expense_analysis",
            "budget_planning",
            "goal_planning",
            "anomaly_detection",
            "health_assessment",
        ]

        for agent in agents:
            mock_structured.invoke.return_value = _IntentResult(
                agent=agent, reasoning=f"Testing {agent}"
            )
            result = classify_intent(f"message for {agent}")
            assert result == agent


# ---------------------------------------------------------------------------
# Integration tests (minimal — just check no crashes)
# ---------------------------------------------------------------------------


class TestIntentClassifierIntegration:
    """Basic integration tests that don't require API key."""

    def test_keyword_fallback_no_crashes(self):
        """Keyword fallback handles various inputs."""
        test_messages = [
            "",
            "hello",
            "hello world" * 100,  # very long
            "🎉 emoji",
            "123 numbers",
        ]
        for msg in test_messages:
            # Should not crash — unknown is valid for out-of-scope messages
            result = _keyword_fallback(msg)
            assert isinstance(result, str)
            assert len(result) > 0
