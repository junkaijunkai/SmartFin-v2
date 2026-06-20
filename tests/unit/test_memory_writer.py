"""
Unit tests for app/memory/writer.py — write_task_memory().

Mock strategy:
- patch app.memory.writer.embed (from app.memory.embeddings import embed)
- patch app.memory.writer.upsert (from app.memory.vector_store import upsert)
- AGENT_SCOPES is NOT mocked — real import from app.orchestrator.state_view

All tests in TestWriteTaskMemory class.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from app.state import BudgetAllocation, FinancialGoal, TransactionCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_goal(name: str, target: float, current: float, target_date: date,
               monthly: float = 500.0, on_track: bool = True) -> FinancialGoal:
    return FinancialGoal(
        id=f"goal-{name.lower().replace(' ', '-')}",
        name=name,
        target_amount=target,
        current_amount=current,
        target_date=target_date,
        required_monthly_saving=monthly,
        on_track=on_track,
    )


def _make_budget_alloc(category: TransactionCategory, allocated: float,
                       spent: float, period_start: date, period_end: date) -> BudgetAllocation:
    return BudgetAllocation(
        category=category,
        allocated_amount=allocated,
        spent_amount=spent,
        period_start=period_start,
        period_end=period_end,
    )


FAKE_EMBEDDING = [0.1] * 1536


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteTaskMemory:
    """Tests for write_task_memory(agent_name, state) in app.memory.writer."""

    # ------------------------------------------------------------------
    # goal_planning — upsert count and id format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_goal_planning_upserts_per_goal(self, mock_embed, mock_upsert):
        """State with 2 goals → upsert called twice, IDs follow goal_planning/{slug} pattern."""
        from app.memory.writer import write_task_memory

        goal1 = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        goal2 = _make_goal("Europe Trip", 5000.0, 1200.0, date(2026, 9, 30))

        state = {"goals": [goal1, goal2]}
        write_task_memory("goal_planning", state)

        assert mock_upsert.call_count == 2

        ids_called = [c.args[0] for c in mock_upsert.call_args_list]
        assert all(id_.startswith("goal_planning/") for id_ in ids_called), \
            f"Expected all ids starting with 'goal_planning/', got: {ids_called}"

        # Slugs should be distinct
        assert ids_called[0] != ids_called[1]

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_goal_planning_ids_match_slugs(self, mock_embed, mock_upsert):
        """Each goal's upsert ID contains a slug derived from the goal name."""
        from app.memory.writer import write_task_memory

        goal1 = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        goal2 = _make_goal("Europe Trip", 5000.0, 1200.0, date(2026, 9, 30))

        state = {"goals": [goal1, goal2]}
        write_task_memory("goal_planning", state)

        ids_called = {c.args[0] for c in mock_upsert.call_args_list}
        assert "goal_planning/emergency-fund" in ids_called
        assert "goal_planning/europe-trip" in ids_called

    # ------------------------------------------------------------------
    # goal_planning — content format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_goal_planning_content_is_natural_language(self, mock_embed, mock_upsert):
        """Upserted content for a goal contains 'targeting' and 'Currently saved'."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31),
                          monthly=500.0, on_track=True)
        state = {"goals": [goal]}
        write_task_memory("goal_planning", state)

        assert mock_upsert.call_count == 1
        content_arg = mock_upsert.call_args.args[2]  # positional: id, memory_type, content, desc, embedding
        assert "targeting" in content_arg, f"Expected 'targeting' in content: {content_arg!r}"
        assert "Currently saved" in content_arg, f"Expected 'Currently saved' in content: {content_arg!r}"

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_goal_planning_content_shows_on_track_status(self, mock_embed, mock_upsert):
        """Content includes 'on track' when goal is on track, 'behind schedule' when not."""
        from app.memory.writer import write_task_memory

        goal_on = _make_goal("Rainy Day", 5000.0, 2000.0, date(2026, 6, 30), on_track=True)
        state = {"goals": [goal_on]}
        write_task_memory("goal_planning", state)
        content_on = mock_upsert.call_args.args[2]
        assert "on track" in content_on

        mock_upsert.reset_mock()

        goal_off = _make_goal("Car Fund", 8000.0, 500.0, date(2026, 6, 30), on_track=False)
        state = {"goals": [goal_off]}
        write_task_memory("goal_planning", state)
        content_off = mock_upsert.call_args.args[2]
        assert "behind schedule" in content_off

    # ------------------------------------------------------------------
    # budget_planning — upsert count and id format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_budget_planning_upserts_one_record(self, mock_embed, mock_upsert):
        """State with budget_allocations + monthly_income → upsert called exactly once."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
            _make_budget_alloc(TransactionCategory.TRANSPORT, 300.0, 120.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        state = {"budget_allocations": allocs, "monthly_income": 5000.0}
        write_task_memory("budget_planning", state)

        assert mock_upsert.call_count == 1

        id_called = mock_upsert.call_args.args[0]
        assert id_called.startswith("budget_planning/"), \
            f"Expected id starting with 'budget_planning/', got: {id_called!r}"

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_budget_planning_id_contains_year_month(self, mock_embed, mock_upsert):
        """Budget upsert ID includes a YYYY-MM component derived from period_start."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        state = {"budget_allocations": allocs, "monthly_income": 5000.0}
        write_task_memory("budget_planning", state)

        id_called = mock_upsert.call_args.args[0]
        assert "2026-06" in id_called, f"Expected '2026-06' in id: {id_called!r}"

    # ------------------------------------------------------------------
    # budget_planning — content format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_budget_planning_content_contains_income(self, mock_embed, mock_upsert):
        """Upserted budget content contains 'Monthly income'."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        state = {"budget_allocations": allocs, "monthly_income": 5000.0}
        write_task_memory("budget_planning", state)

        content_arg = mock_upsert.call_args.args[2]
        assert "Monthly income" in content_arg, \
            f"Expected 'Monthly income' in content: {content_arg!r}"

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_budget_planning_content_shows_over_budget(self, mock_embed, mock_upsert):
        """Content mentions over-budget categories (spent > allocated)."""
        from app.memory.writer import write_task_memory

        allocs = [
            # Over budget: spent 900 vs 800 limit
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 900.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        state = {"budget_allocations": allocs, "monthly_income": 5000.0}
        write_task_memory("budget_planning", state)

        content_arg = mock_upsert.call_args.args[2]
        # Over-budget entry should appear with a "+" indicator
        assert "+" in content_arg or "over" in content_arg.lower(), \
            f"Expected over-budget indicator in content: {content_arg!r}"

    # ------------------------------------------------------------------
    # No memory fields → skip upsert
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_no_memory_fields_skips_upsert(self, mock_embed, mock_upsert):
        """expense_analysis has memory=set() → upsert never called."""
        from app.memory.writer import write_task_memory

        state = {
            "categorised_transactions": [],
            "spending_trends": [],
        }
        write_task_memory("expense_analysis", state)

        mock_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Empty goals list → skip upsert
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_empty_goals_skips_upsert(self, mock_embed, mock_upsert):
        """goals=[] → upsert not called (nothing to persist)."""
        from app.memory.writer import write_task_memory

        state = {"goals": []}
        write_task_memory("goal_planning", state)

        mock_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Failure silence — embed raises
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", side_effect=RuntimeError("embedding API down"))
    def test_embed_failure_is_silent(self, mock_embed, mock_upsert):
        """If embed() raises, write_task_memory() does not propagate the exception."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        state = {"goals": [goal]}

        # Should NOT raise
        write_task_memory("goal_planning", state)

        # upsert should not be called since embed failed
        mock_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Failure silence — upsert raises
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert", side_effect=Exception("DB connection error"))
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_upsert_failure_is_silent(self, mock_embed, mock_upsert):
        """If upsert() raises, write_task_memory() does not propagate the exception."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        state = {"goals": [goal]}

        # Should NOT raise
        write_task_memory("goal_planning", state)

    # ------------------------------------------------------------------
    # embed called with text that contains key goal fields
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_embed_receives_goal_text(self, mock_embed, mock_upsert):
        """embed() is called with text describing the goal (not raw JSON)."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        state = {"goals": [goal]}
        write_task_memory("goal_planning", state)

        assert mock_embed.call_count == 1
        embed_text = mock_embed.call_args.args[0]
        # Verify it's natural language, not raw dict/JSON
        assert "Emergency Fund" in embed_text, \
            f"Expected goal name in embed text: {embed_text!r}"
        assert "{" not in embed_text or "targeting" in embed_text, \
            f"embed text looks like raw JSON: {embed_text!r}"

    # ------------------------------------------------------------------
    # description field is a short one-liner
    # ------------------------------------------------------------------

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_goal_description_is_short_summary(self, mock_embed, mock_upsert):
        """The description arg to upsert is a compact one-line summary."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31),
                          on_track=True)
        state = {"goals": [goal]}
        write_task_memory("goal_planning", state)

        # 4th positional arg is description
        desc_arg = mock_upsert.call_args.args[3]
        assert "Emergency Fund" in desc_arg, f"Goal name missing in description: {desc_arg!r}"
        assert "\n" not in desc_arg, f"Description should be one line: {desc_arg!r}"

    @patch("app.memory.writer.upsert")
    @patch("app.memory.writer.embed", return_value=FAKE_EMBEDDING)
    def test_budget_description_is_short_summary(self, mock_embed, mock_upsert):
        """Budget upsert description is a compact one-liner."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        state = {"budget_allocations": allocs, "monthly_income": 5000.0}
        write_task_memory("budget_planning", state)

        desc_arg = mock_upsert.call_args.args[3]
        assert "Budget" in desc_arg or "budget" in desc_arg, \
            f"Expected 'Budget' in description: {desc_arg!r}"
        assert "\n" not in desc_arg, f"Description should be one line: {desc_arg!r}"
