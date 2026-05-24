"""
Unit tests for the Expense Analysis agent.

Categoriser tests use unittest.mock to patch ChatAnthropic so no real API
calls are made.  Analyser tests are pure Python — no mocking needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.state import (
    AppState,
    Transaction,
    TransactionCategory,
)
from app.agents.expense_analysis.analyser import PERIOD_DAYS, compute_spending_trends
from app.agents.expense_analysis.categoriser import CHUNK_SIZE, MAX_RETRIES, categorise_transactions
from app.agents.expense_analysis.agent import run as expense_analysis_run
from app.orchestrator.state_view import AgentStateView


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tx(
    id_: str,
    days_ago: float,
    amount: float,
    category: TransactionCategory = TransactionCategory.OTHER,
) -> Transaction:
    """Create a Transaction with a date relative to now."""
    return Transaction(
        id=id_,
        date=datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
        amount=amount,
        description="test transaction",
        merchant="Test Merchant",
        category=category,
    )


def _mock_chain(id_category_pairs: list[tuple[str, TransactionCategory]]) -> MagicMock:
    """Return a mock structured-output chain that yields the given classifications."""
    results = [
        SimpleNamespace(transaction_id=id_, category=cat)
        for id_, cat in id_category_pairs
    ]
    chain = MagicMock()
    chain.invoke.return_value = SimpleNamespace(results=results)
    return chain


# ---------------------------------------------------------------------------
# Analyser tests  (pure Python, no LLM)
# ---------------------------------------------------------------------------


class TestComputeSpendingTrends:

    def test_basic_deviation_calculation(self):
        txs = [
            _tx("a1", 5, 100.0, TransactionCategory.FOOD),   # current period
            _tx("a2", 35, 80.0, TransactionCategory.FOOD),   # previous period
        ]
        trends = compute_spending_trends(txs)
        food = next(t for t in trends if t.category == TransactionCategory.FOOD)
        assert food.current_period_total == 100.0
        assert food.previous_period_total == 80.0
        assert food.deviation_pct == pytest.approx(25.0)

    def test_decrease_gives_negative_deviation(self):
        txs = [
            _tx("b1", 5, 60.0, TransactionCategory.TRANSPORT),   # current
            _tx("b2", 35, 100.0, TransactionCategory.TRANSPORT),  # previous
        ]
        trends = compute_spending_trends(txs)
        t = trends[0]
        assert t.deviation_pct == pytest.approx(-40.0)

    def test_no_previous_period_returns_none(self):
        """Category appearing only in the current period → deviation_pct is None."""
        txs = [_tx("c1", 5, 50.0, TransactionCategory.HEALTHCARE)]
        trends = compute_spending_trends(txs)
        assert len(trends) == 1
        assert trends[0].deviation_pct is None

    def test_no_current_period_returns_none(self):
        """Category only in the previous period also has deviation_pct = None."""
        txs = [_tx("d1", 45, 50.0, TransactionCategory.EDUCATION)]
        trends = compute_spending_trends(txs)
        assert trends[0].deviation_pct is None

    def test_income_excluded_from_trends(self):
        """Negative amounts (income) must not appear in spending trends."""
        txs = [_tx("e1", 5, -3000.0, TransactionCategory.INCOME)]
        assert compute_spending_trends(txs) == []

    def test_sorted_by_current_period_descending(self):
        txs = [
            _tx("f1", 5,  50.0, TransactionCategory.FOOD),
            _tx("f2", 5, 200.0, TransactionCategory.HOUSING),
            _tx("f3", 5, 100.0, TransactionCategory.TRANSPORT),
        ]
        trends = compute_spending_trends(txs)
        totals = [t.current_period_total for t in trends]
        assert totals == sorted(totals, reverse=True)

    def test_empty_input(self):
        assert compute_spending_trends([]) == []

    def test_boundary_transaction_at_period_edge(self):
        """A transaction exactly at PERIOD_DAYS ago falls in the current period."""
        txs = [_tx("g1", PERIOD_DAYS - 0.01, 75.0, TransactionCategory.SHOPPING)]
        trends = compute_spending_trends(txs)
        assert trends[0].current_period_total == 75.0
        assert trends[0].previous_period_total == 0.0


# ---------------------------------------------------------------------------
# Categoriser tests  (mock ChatAnthropic)
# ---------------------------------------------------------------------------


class TestCategoriseTransactions:

    def test_categories_are_assigned(self):
        txs = [
            Transaction(id="x1", date=datetime.now(tz=timezone.utc), amount=50.0,
                        description="Tesco shop", merchant="Tesco"),
            Transaction(id="x2", date=datetime.now(tz=timezone.utc), amount=20.0,
                        description="Bus pass", merchant="TfL"),
        ]
        chain = _mock_chain([
            ("x1", TransactionCategory.FOOD),
            ("x2", TransactionCategory.TRANSPORT),
        ])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, llm_ok = categorise_transactions(txs)

        assert result[0].category == TransactionCategory.FOOD
        assert result[1].category == TransactionCategory.TRANSPORT
        assert llm_ok is True

    def test_missing_id_defaults_to_other(self):
        """If the LLM omits a transaction id, that transaction falls back to OTHER."""
        txs = [
            Transaction(id="y1", date=datetime.now(tz=timezone.utc), amount=30.0,
                        description="Mystery charge", merchant="Unknown"),
        ]
        chain = _mock_chain([])  # LLM returns no results
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, _ = categorise_transactions(txs)

        assert result[0].category == TransactionCategory.OTHER

    def test_empty_input_returns_empty(self):
        result, llm_ok = categorise_transactions([])
        assert result == []
        assert llm_ok is True

    def test_original_transactions_not_mutated(self):
        """categorise_transactions must return new objects; originals stay unchanged."""
        tx = Transaction(id="z1", date=datetime.now(tz=timezone.utc), amount=10.0,
                         description="Coffee", merchant="Cafe")
        chain = _mock_chain([("z1", TransactionCategory.FOOD)])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, _ = categorise_transactions([tx])

        assert tx.category == TransactionCategory.OTHER   # original untouched
        assert result[0].category == TransactionCategory.FOOD

    def test_llm_called_once_for_any_batch_size(self):
        """Only one LLM call is made regardless of how many transactions are passed."""
        txs = [
            Transaction(id=f"m{i}", date=datetime.now(tz=timezone.utc), amount=10.0,
                        description="tx", merchant="M")
            for i in range(10)
        ]
        chain = _mock_chain([(f"m{i}", TransactionCategory.OTHER) for i in range(10)])
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            categorise_transactions(txs)

        chain.invoke.assert_called_once()

    def test_retries_on_failure_then_succeeds(self):
        """LLM succeeds on the second attempt; invoke is called twice."""
        txs = [Transaction(id="r1", date=datetime.now(tz=timezone.utc), amount=10.0,
                           description="Coffee", merchant="Cafe")]
        chain = MagicMock()
        chain.invoke.side_effect = [
            RuntimeError("transient error"),   # attempt 1 fails
            SimpleNamespace(results=[          # attempt 2 succeeds
                SimpleNamespace(transaction_id="r1", category=TransactionCategory.FOOD)
            ]),
        ]
        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is True
        assert result[0].category == TransactionCategory.FOOD
        assert chain.invoke.call_count == 2

    def test_falls_back_after_max_retries(self):
        """All LLM attempts fail → keyword fallback used, llm_ok=False."""
        txs = [Transaction(id="fb1", date=datetime.now(tz=timezone.utc), amount=50.0,
                           description="Weekly grocery shop", merchant="Tesco")]
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("API down")

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is False
        assert chain.invoke.call_count == 3  # MAX_RETRIES
        assert result[0].category == TransactionCategory.FOOD  # keyword matched "tesco"

    def test_fallback_flags_income_by_negative_amount(self):
        """Fallback must classify negative amounts as income regardless of description."""
        txs = [Transaction(id="inc1", date=datetime.now(tz=timezone.utc), amount=-3200.0,
                           description="Monthly salary", merchant="Employer Ltd")]
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("API down")

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is False
        assert result[0].category == TransactionCategory.INCOME

    def test_large_input_is_split_into_chunks(self):
        """CHUNK_SIZE + 1 transactions must trigger two separate LLM invoke calls."""
        txs = [
            Transaction(id=f"c{i}", date=datetime.now(tz=timezone.utc), amount=10.0,
                        description="tx", merchant="M")
            for i in range(CHUNK_SIZE + 1)
        ]
        # Return valid results for every id in both calls
        def make_batch(chunk_txs):
            return SimpleNamespace(results=[
                SimpleNamespace(transaction_id=t.id, category=TransactionCategory.OTHER)
                for t in chunk_txs
            ])

        chain = MagicMock()
        chain.invoke.side_effect = [
            make_batch(txs[:CHUNK_SIZE]),
            make_batch(txs[CHUNK_SIZE:]),
        ]

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, llm_ok = categorise_transactions(txs)

        assert chain.invoke.call_count == 2
        assert len(result) == CHUNK_SIZE + 1
        assert llm_ok is True

    def test_chunked_results_preserve_original_order(self):
        """Output transactions must appear in the same order as the input."""
        txs = [
            Transaction(id=f"o{i}", date=datetime.now(tz=timezone.utc), amount=float(i),
                        description="tx", merchant="M")
            for i in range(CHUNK_SIZE + 5)
        ]
        def make_batch(chunk_txs):
            return SimpleNamespace(results=[
                SimpleNamespace(transaction_id=t.id, category=TransactionCategory.OTHER)
                for t in chunk_txs
            ])

        chain = MagicMock()
        chain.invoke.side_effect = [
            make_batch(txs[:CHUNK_SIZE]),
            make_batch(txs[CHUNK_SIZE:]),
        ]

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            result, _ = categorise_transactions(txs)

        assert [t.id for t in result] == [t.id for t in txs]

    def test_partial_chunk_failure_sets_llm_ok_false(self):
        """If one chunk exhausts retries, llm_succeeded is False for the whole batch."""
        txs = [
            Transaction(id=f"p{i}", date=datetime.now(tz=timezone.utc), amount=10.0,
                        description="tx", merchant="M")
            for i in range(CHUNK_SIZE + 1)
        ]
        good_batch = SimpleNamespace(results=[
            SimpleNamespace(transaction_id=t.id, category=TransactionCategory.OTHER)
            for t in txs[:CHUNK_SIZE]
        ])

        chain = MagicMock()
        # First chunk succeeds; second chunk fails all retries
        chain.invoke.side_effect = [good_batch] + [RuntimeError("API down")] * MAX_RETRIES

        with patch("app.agents.expense_analysis.categoriser.ChatAnthropic") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value = chain
            with patch("app.agents.expense_analysis.categoriser.time.sleep"):
                result, llm_ok = categorise_transactions(txs)

        assert llm_ok is False
        assert len(result) == CHUNK_SIZE + 1  # all transactions still returned


# ---------------------------------------------------------------------------
# Agent node test  (mock run_react_loop, test pre/post processing)
# ---------------------------------------------------------------------------


class TestExpenseAnalysisAgentNode:
    """Tests for the agent node function with ReAct loop mocked."""

    def _run(self, state: dict) -> dict:
        return expense_analysis_run(AgentStateView(state, "expense_analysis"))

    def _make_state(self, transactions: list[Transaction]) -> AppState:
        return {
            "messages": [],
            "transactions": transactions,
            "monthly_income": 3000.0,
            "categorised_transactions": [],
            "spending_trends": [],
            "budget_allocations": [],
            "goals": [],
            "anomaly_flags": [],
            "health_summary": None,
            "alerts": [],
            "pending_confirmation": None,
            "active_agent": "expense_analysis",
            "pending_intent": None,
        }

    def _mock_react(self, tool_ctx: dict) -> ...:
        """Return a side-effect function that populates tool_ctx for run_react_loop."""
        from unittest.mock import patch

        def side_effect(*, tool_ctx= None, **kwargs):
            if tool_ctx is not None:
                tool_ctx.update(tool_ctx)
            return Mock(), []

        return patch(
            "app.agents.expense_analysis.agent.run_react_loop",
            side_effect=lambda *a, tool_ctx=tc, **kw: (
                tool_ctx.update(tc) if tc else None,
                []
            )[-1],
        )

    def test_writes_required_state_fields(self):
        txs = [
            _tx("e1", 5, 100.0, TransactionCategory.FOOD),
            _tx("e2", 35, 80.0, TransactionCategory.FOOD),
        ]
        from unittest.mock import Mock, patch

        categorised = [
            _tx("e1", 5, 100.0, TransactionCategory.FOOD),
            _tx("e2", 35, 80.0, TransactionCategory.FOOD),
        ]
        trends = compute_spending_trends(categorised)

        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    tool_ctx["categorised"] = categorised
                    tool_ctx["trends"] = trends
                    tool_ctx["summary"] = "Categorised 2 transactions."
                    tool_ctx["needs_hitl"] = True
                    tool_ctx["categorise_llm_ok"] = True
                return Mock(), []

            mock_loop.side_effect = side
            result = self._run(self._make_state(txs))

        assert len(result["categorised_transactions"]) == 2
        assert len(result["spending_trends"]) == 1
        assert result["pending_confirmation"]["action"] == "approve_expense_analysis"
        assert result["pending_confirmation"]["categorisation_confidence"] == "llm"

    def test_hitl_payload_has_no_confirmed_key(self):
        from unittest.mock import Mock, patch

        txs = [_tx("h1", 5, 50.0, TransactionCategory.SHOPPING)]
        trends = compute_spending_trends(txs)

        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    tool_ctx["categorised"] = txs
                    tool_ctx["trends"] = trends
                    tool_ctx["needs_hitl"] = True
                    tool_ctx["categorise_llm_ok"] = True
                return Mock(), []

            mock_loop.side_effect = side
            result = self._run(self._make_state(txs))

        assert "confirmed" not in result["pending_confirmation"]

    def test_empty_transactions_skips_loop(self):
        """With no transactions at all, the agent should return empty results
        without invoking the ReAct loop."""
        from unittest.mock import patch

        with patch(
            "app.agents.expense_analysis.agent.run_react_loop"
        ) as mock_loop:
            result = expense_analysis_run(self._make_state([]))

        mock_loop.assert_not_called()
        assert result["categorised_transactions"] == []
        assert result["spending_trends"] == []

    # =========================================================================
    # Incremental processing tests  (pre-processing logic, no ReAct loop needed)
    # =========================================================================

    def test_incremental_only_new_transactions_no_existing(self):
        """First run: new transactions, no existing categorised data."""
        txs = [_tx("n1", 5, 100.0, TransactionCategory.FOOD)]
        from unittest.mock import Mock, patch

        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    tool_ctx["categorised"] = txs
                    tool_ctx["trends"] = compute_spending_trends(txs)
                    tool_ctx["summary"] = "Categorised 1 transaction."
                    tool_ctx["needs_hitl"] = True
                    tool_ctx["categorise_llm_ok"] = True
                return Mock(), []

            mock_loop.side_effect = side
            state = self._make_state(txs)
            state["categorised_transactions"] = []
            result = self._run(state)

        assert len(result["categorised_transactions"]) == 1
        assert result["categorised_transactions"][0].id == "n1"
        assert len(result["spending_trends"]) == 1

    def test_incremental_only_existing_data_no_new_transactions(self):
        """Second run: only existing categorised data, no new txs to process."""
        existing_tx = _tx("e1", 5, 100.0, TransactionCategory.FOOD)
        state = self._make_state([])
        state["categorised_transactions"] = [existing_tx]
        result = self._run(state)

        assert len(result["categorised_transactions"]) == 1
        assert result["categorised_transactions"][0].id == "e1"
        assert len(result["spending_trends"]) == 1

    def test_incremental_merge_existing_and_new_transactions(self):
        from unittest.mock import Mock, patch

        existing = _tx("e1", 5, 100.0, TransactionCategory.FOOD)
        new_tx = _tx("n1", 5, 50.0, TransactionCategory.FOOD)

        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    merged = [existing] + [_tx("n1", 5, 50.0, TransactionCategory.FOOD)]
                    tool_ctx["categorised"] = merged
                    tool_ctx["trends"] = compute_spending_trends(merged)
                return Mock(), []

            mock_loop.side_effect = side

            state = self._make_state([new_tx])
            state["categorised_transactions"] = [existing]
            result = self._run(state)

        assert len(result["categorised_transactions"]) == 2
        ids = {t.id for t in result["categorised_transactions"]}
        assert ids == {"e1", "n1"}

    def test_incremental_trends_calculated_on_merged_data(self):
        from unittest.mock import Mock, patch

        existing_tx1 = _tx("e1", 5, 100.0, TransactionCategory.FOOD)
        existing_tx2 = _tx("e2", 35, 80.0, TransactionCategory.FOOD)
        new_tx = _tx("n1", 5, 50.0, TransactionCategory.FOOD)

        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    merged = [existing_tx1, existing_tx2] + [_tx("n1", 5, 50.0, TransactionCategory.FOOD)]
                    tool_ctx["categorised"] = merged
                    tool_ctx["trends"] = compute_spending_trends(merged)
                return Mock(), []

            mock_loop.side_effect = side

            state = self._make_state([new_tx])
            state["categorised_transactions"] = [existing_tx1, existing_tx2]
            result = self._run(state)

        trend = result["spending_trends"][0]
        assert trend.category == TransactionCategory.FOOD
        assert trend.current_period_total == 150.0
        assert trend.previous_period_total == 80.0
        assert trend.deviation_pct == pytest.approx(87.5)

    def test_incremental_filters_duplicate_transaction_ids(self):
        existing_tx = _tx("dup1", 5, 100.0, TransactionCategory.FOOD)
        from unittest.mock import patch

        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            state = self._make_state([existing_tx])
            state["categorised_transactions"] = [existing_tx]
            result = self._run(state)

        # No new transactions to process — ReAct loop not called
        mock_loop.assert_not_called()
        assert len(result["categorised_transactions"]) == 1
        assert result["categorised_transactions"][0].id == "dup1"

    def test_expense_analysis_dict_contains_monthly_avg_and_trends(self):
        from unittest.mock import Mock, patch

        txs = [
            _tx("e1", 5, 100.0, TransactionCategory.FOOD),
            _tx("e2", 35, 80.0, TransactionCategory.FOOD),
        ]
        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    tool_ctx["categorised"] = txs
                    tool_ctx["trends"] = compute_spending_trends(txs)
                    tool_ctx["categorise_llm_ok"] = True
                    tool_ctx["needs_hitl"] = True
                return Mock(), []

            mock_loop.side_effect = side
            result = self._run(self._make_state(txs))

        assert "expense_analysis" in result
        assert "category_monthly_avg" in result["expense_analysis"]
        assert "category_trends" in result["expense_analysis"]

    def test_category_trends_classifies_deviation_correctly(self):
        from unittest.mock import Mock, patch

        txs = [
            _tx("t1", 5, 110.0, TransactionCategory.FOOD),
            _tx("t2", 5, 90.0, TransactionCategory.TRANSPORT),
            _tx("t3", 5, 105.0, TransactionCategory.SHOPPING),
            _tx("t4", 5, 50.0, TransactionCategory.HEALTHCARE),
            _tx("t1p", 35, 100.0, TransactionCategory.FOOD),
            _tx("t2p", 35, 100.0, TransactionCategory.TRANSPORT),
            _tx("t3p", 35, 100.0, TransactionCategory.SHOPPING),
        ]
        with patch("app.agents.expense_analysis.agent.run_react_loop") as mock_loop:

            def side(tool_ctx=None, **kw):
                if tool_ctx is not None:
                    tool_ctx["categorised"] = txs
                    tool_ctx["trends"] = compute_spending_trends(txs)
                    tool_ctx["categorise_llm_ok"] = True
                    tool_ctx["needs_hitl"] = True
                return Mock(), []

            mock_loop.side_effect = side
            result = self._run(self._make_state(txs))

        trends = result["expense_analysis"]["category_trends"]
        assert trends["food"] == "rising"
        assert trends["transport"] == "volatile"
        assert trends["shopping"] == "stable"
        assert trends["healthcare"] == "fixed"
