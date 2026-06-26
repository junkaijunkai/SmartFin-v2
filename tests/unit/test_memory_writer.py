"""
Unit tests for app/memory/writer.py — write_task_memory().

Mock strategy:
- patch app.memory.writer.fetch_hashes_batch → returns {} (treats every record as new)
- patch app.memory.writer.embed_batch → returns [FAKE_EMBEDDING, ...]
- patch app.memory.writer.batch_upsert → records captured for assertions
- AGENT_SCOPES is NOT mocked — real import from app.orchestrator.state_view

All tests in TestWriteTaskMemory class.
"""

from __future__ import annotations

import hashlib
from datetime import date
from unittest.mock import MagicMock, patch

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


def _mock_embed_batch(texts):
    """Return one fake embedding per input text."""
    return [FAKE_EMBEDDING[:] for _ in texts]


# ---------------------------------------------------------------------------
# Helpers to extract records from batch_upsert calls
# ---------------------------------------------------------------------------


def _upserted_records(mock_batch_upsert) -> list[dict]:
    """Flatten all records passed across all batch_upsert calls."""
    records = []
    for call in mock_batch_upsert.call_args_list:
        records.extend(call.args[0])
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteTaskMemory:
    """Tests for write_task_memory(agent_name, state) in app.memory.writer."""

    # ------------------------------------------------------------------
    # goal_planning — upsert count and id format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_planning_upserts_per_goal(self, mock_fetch, mock_embed, mock_batch_upsert):
        """State with 2 goals → batch_upsert called with 2 records, IDs follow goal_planning/{slug}."""
        from app.memory.writer import write_task_memory

        goal1 = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        goal2 = _make_goal("Europe Trip", 5000.0, 1200.0, date(2026, 9, 30))

        write_task_memory("goal_planning", {"goals": [goal1, goal2]})

        records = _upserted_records(mock_batch_upsert)
        assert len(records) == 2

        ids = [r["id"] for r in records]
        assert all(id_.startswith("goal_planning/") for id_ in ids), \
            f"Expected all ids starting with 'goal_planning/', got: {ids}"
        assert ids[0] != ids[1]

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_planning_ids_match_slugs(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Each goal's record ID contains a slug derived from the goal name."""
        from app.memory.writer import write_task_memory

        goal1 = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        goal2 = _make_goal("Europe Trip", 5000.0, 1200.0, date(2026, 9, 30))

        write_task_memory("goal_planning", {"goals": [goal1, goal2]})

        ids = {r["id"] for r in _upserted_records(mock_batch_upsert)}
        assert "goal_planning/emergency-fund" in ids
        assert "goal_planning/europe-trip" in ids

    # ------------------------------------------------------------------
    # goal_planning — content format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_planning_content_is_natural_language(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Upserted content for a goal contains 'targeting' and 'Currently saved'."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31),
                          monthly=500.0, on_track=True)
        write_task_memory("goal_planning", {"goals": [goal]})

        records = _upserted_records(mock_batch_upsert)
        assert len(records) == 1
        content = records[0]["content"]
        assert "targeting" in content, f"Expected 'targeting' in content: {content!r}"
        assert "Currently saved" in content, f"Expected 'Currently saved' in content: {content!r}"

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_planning_content_shows_on_track_status(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Content includes 'on track' when goal is on track, 'behind schedule' when not."""
        from app.memory.writer import write_task_memory

        goal_on = _make_goal("Rainy Day", 5000.0, 2000.0, date(2026, 6, 30), on_track=True)
        write_task_memory("goal_planning", {"goals": [goal_on]})
        content_on = _upserted_records(mock_batch_upsert)[0]["content"]
        assert "on track" in content_on

        mock_batch_upsert.reset_mock()

        goal_off = _make_goal("Car Fund", 8000.0, 500.0, date(2026, 6, 30), on_track=False)
        write_task_memory("goal_planning", {"goals": [goal_off]})
        content_off = _upserted_records(mock_batch_upsert)[0]["content"]
        assert "behind schedule" in content_off

    # ------------------------------------------------------------------
    # budget_planning — upsert count and id format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_budget_planning_upserts_one_record(self, mock_fetch, mock_embed, mock_batch_upsert):
        """State with budget_allocations + monthly_income → batch_upsert called with exactly one record."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
            _make_budget_alloc(TransactionCategory.TRANSPORT, 300.0, 120.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        write_task_memory("budget_planning", {"budget_allocations": allocs, "monthly_income": 5000.0})

        records = _upserted_records(mock_batch_upsert)
        assert len(records) == 1
        assert records[0]["id"].startswith("budget_planning/"), \
            f"Expected id starting with 'budget_planning/', got: {records[0]['id']!r}"

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_budget_planning_id_contains_year_month(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Budget record ID includes a YYYY-MM component derived from period_start."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        write_task_memory("budget_planning", {"budget_allocations": allocs, "monthly_income": 5000.0})

        record_id = _upserted_records(mock_batch_upsert)[0]["id"]
        assert "2026-06" in record_id, f"Expected '2026-06' in id: {record_id!r}"

    # ------------------------------------------------------------------
    # budget_planning — content format
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_budget_planning_content_contains_income(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Upserted budget content contains 'Monthly income'."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        write_task_memory("budget_planning", {"budget_allocations": allocs, "monthly_income": 5000.0})

        content = _upserted_records(mock_batch_upsert)[0]["content"]
        assert "Monthly income" in content, \
            f"Expected 'Monthly income' in content: {content!r}"

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_budget_planning_content_shows_over_budget(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Content mentions over-budget categories (spent > allocated)."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 900.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        write_task_memory("budget_planning", {"budget_allocations": allocs, "monthly_income": 5000.0})

        content = _upserted_records(mock_batch_upsert)[0]["content"]
        assert "+" in content or "over" in content.lower(), \
            f"Expected over-budget indicator in content: {content!r}"

    # ------------------------------------------------------------------
    # No memory fields → skip upsert
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_no_memory_fields_skips_upsert(self, mock_fetch, mock_embed, mock_batch_upsert):
        """expense_analysis has memory=set() → batch_upsert never called."""
        from app.memory.writer import write_task_memory

        write_task_memory("expense_analysis", {"categorised_transactions": [], "spending_trends": []})

        mock_batch_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Empty goals list → skip upsert
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_empty_goals_skips_upsert(self, mock_fetch, mock_embed, mock_batch_upsert):
        """goals=[] → batch_upsert not called (nothing to persist)."""
        from app.memory.writer import write_task_memory

        write_task_memory("goal_planning", {"goals": []})

        mock_batch_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Unchanged content → skip embed and upsert
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch")
    def test_unchanged_goal_skips_embed(self, mock_fetch_hashes, mock_embed, mock_batch_upsert):
        """If stored hash matches candidate hash, embed_batch and batch_upsert are not called."""
        from app.memory.writer import write_task_memory, _goal_to_text, _safe_slug

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        stored_text = _goal_to_text(goal)
        stored_hash = hashlib.sha256(stored_text.encode()).hexdigest()
        record_id = f"goal_planning/{_safe_slug(goal.name)}"
        mock_fetch_hashes.return_value = {record_id: stored_hash}

        write_task_memory("goal_planning", {"goals": [goal]})

        mock_embed.assert_not_called()
        mock_batch_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # dirty_fields parameter — skip processing when field not dirtied
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_dirty_fields_skips_when_not_in_scope(self, mock_fetch, mock_embed, mock_batch_upsert):
        """dirty_fields provided but doesn't include any memory field → upsert not called."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        # dirty_fields contains no memory-scoped field for goal_planning (only "goals" is)
        write_task_memory("goal_planning", {"goals": [goal]}, dirty_fields={"messages"})

        mock_batch_upsert.assert_not_called()

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_dirty_fields_triggers_when_in_scope(self, mock_fetch, mock_embed, mock_batch_upsert):
        """dirty_fields includes 'goals' → upsert is called."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        write_task_memory("goal_planning", {"goals": [goal]}, dirty_fields={"goals"})

        records = _upserted_records(mock_batch_upsert)
        assert len(records) == 1

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_dirty_fields_none_processes_all(self, mock_fetch, mock_embed, mock_batch_upsert):
        """dirty_fields=None (HITL/legacy path) processes all memory-scoped fields."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        write_task_memory("goal_planning", {"goals": [goal]}, dirty_fields=None)

        records = _upserted_records(mock_batch_upsert)
        assert len(records) == 1

    # ------------------------------------------------------------------
    # Failure silence — embed_batch raises
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=RuntimeError("embedding API down"))
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_embed_failure_is_silent(self, mock_fetch, mock_embed, mock_batch_upsert):
        """If embed_batch() raises, write_task_memory() does not propagate the exception."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        write_task_memory("goal_planning", {"goals": [goal]})

        mock_batch_upsert.assert_not_called()

    # ------------------------------------------------------------------
    # Failure silence — batch_upsert raises
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert", side_effect=Exception("DB connection error"))
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_upsert_failure_is_silent(self, mock_fetch, mock_embed, mock_batch_upsert):
        """If batch_upsert() raises, write_task_memory() does not propagate the exception."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        write_task_memory("goal_planning", {"goals": [goal]})  # should NOT raise

    # ------------------------------------------------------------------
    # embed_batch called with text that contains key goal fields
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_embed_receives_goal_text(self, mock_fetch, mock_embed, mock_batch_upsert):
        """embed_batch() is called with a list containing natural-language goal text."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        write_task_memory("goal_planning", {"goals": [goal]})

        assert mock_embed.call_count == 1
        texts = mock_embed.call_args.args[0]
        assert isinstance(texts, list) and len(texts) == 1
        embed_text = texts[0]
        assert "Emergency Fund" in embed_text, \
            f"Expected goal name in embed text: {embed_text!r}"
        assert "{" not in embed_text or "targeting" in embed_text, \
            f"embed text looks like raw JSON: {embed_text!r}"

    # ------------------------------------------------------------------
    # description field is a short one-liner
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_description_is_short_summary(self, mock_fetch, mock_embed, mock_batch_upsert):
        """The description in the upserted record is a compact one-line summary."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31), on_track=True)
        write_task_memory("goal_planning", {"goals": [goal]})

        record = _upserted_records(mock_batch_upsert)[0]
        desc = record["description"]
        assert "Emergency Fund" in desc, f"Goal name missing in description: {desc!r}"
        assert "\n" not in desc, f"Description should be one line: {desc!r}"

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_budget_description_is_short_summary(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Budget record description is a compact one-liner."""
        from app.memory.writer import write_task_memory

        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               date(2026, 6, 1), date(2026, 6, 30)),
        ]
        write_task_memory("budget_planning", {"budget_allocations": allocs, "monthly_income": 5000.0})

        record = _upserted_records(mock_batch_upsert)[0]
        desc = record["description"]
        assert "Budget" in desc or "budget" in desc, \
            f"Expected 'Budget' in description: {desc!r}"
        assert "\n" not in desc, f"Description should be one line: {desc!r}"

    # ------------------------------------------------------------------
    # expires_at is set on records
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_record_has_expires_at(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Goal records include a non-None expires_at derived from target_date + 30 days."""
        from app.memory.writer import write_task_memory
        from datetime import datetime, timezone

        target = date(2026, 12, 31)
        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, target)
        write_task_memory("goal_planning", {"goals": [goal]})

        record = _upserted_records(mock_batch_upsert)[0]
        expires_at = record["expires_at"]
        assert expires_at is not None, "Goal record should have expires_at set"
        assert isinstance(expires_at, datetime)
        # expires_at should be after target_date
        assert expires_at.date() > target

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_budget_record_has_expires_at(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Budget records include a non-None expires_at ~90 days after period_start."""
        from app.memory.writer import write_task_memory
        from datetime import datetime, timedelta

        period_start = date(2026, 6, 1)
        allocs = [
            _make_budget_alloc(TransactionCategory.FOOD, 800.0, 650.0,
                               period_start, date(2026, 6, 30)),
        ]
        write_task_memory("budget_planning", {"budget_allocations": allocs, "monthly_income": 5000.0})

        record = _upserted_records(mock_batch_upsert)[0]
        expires_at = record["expires_at"]
        assert expires_at is not None, "Budget record should have expires_at set"
        assert isinstance(expires_at, datetime)
        # Should be approximately 90 days after period_start
        expected = datetime.combine(period_start, datetime.min.time()).replace(microsecond=0)
        delta = abs((expires_at.replace(tzinfo=None) - expected).days - 90)
        assert delta <= 1, f"Expected ~90 days from period_start, got {(expires_at.replace(tzinfo=None) - expected).days} days"

    # ------------------------------------------------------------------
    # content_hash is stored in upserted records
    # ------------------------------------------------------------------

    @patch("app.memory.writer.batch_upsert")
    @patch("app.memory.writer.embed_batch", side_effect=_mock_embed_batch)
    @patch("app.memory.writer.fetch_hashes_batch", return_value={})
    def test_goal_record_has_content_hash(self, mock_fetch, mock_embed, mock_batch_upsert):
        """Upserted goal record includes a content_hash that matches SHA-256 of the content."""
        from app.memory.writer import write_task_memory

        goal = _make_goal("Emergency Fund", 10000.0, 3000.0, date(2026, 12, 31))
        write_task_memory("goal_planning", {"goals": [goal]})

        record = _upserted_records(mock_batch_upsert)[0]
        assert "content_hash" in record, "Record should include content_hash"
        expected_hash = hashlib.sha256(record["content"].encode()).hexdigest()
        assert record["content_hash"] == expected_hash, \
            f"content_hash mismatch: {record['content_hash']!r} != {expected_hash!r}"
