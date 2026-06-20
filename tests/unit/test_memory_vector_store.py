"""
Unit tests for memory vector store layer.

Tests cover embedding generation and vector store CRUD operations.
All external services (OpenAI, psycopg) are mocked.
"""

from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Tests for embed()
# ---------------------------------------------------------------------------


class TestEmbed:
    """Test OpenAI embedding generation."""

    @patch("app.memory.embeddings._get_client")
    def test_embed_returns_list_of_floats(self, mock_get_client):
        """embed() returns a list of 1536 floats on success."""
        from app.memory.embeddings import embed

        # Build mock response
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        fake_embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding)]
        mock_client.embeddings.create.return_value = mock_response

        result = embed("hello")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)

    @patch("app.memory.embeddings._get_client")
    def test_embed_propagates_exception(self, mock_get_client):
        """embed() does not swallow exceptions from the OpenAI client."""
        from app.memory.embeddings import embed

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            embed("hello")


# ---------------------------------------------------------------------------
# Tests for VectorStore operations
# ---------------------------------------------------------------------------


class TestVectorStore:
    """Test psycopg-backed vector store operations."""

    def _make_mock_conn(self, fetchall_return=None):
        """Helper: build a mock psycopg connection context manager."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = fetchall_return or []

        # Support `with psycopg.connect(...) as conn:` and `with conn.cursor() as cur:`
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        return mock_conn, mock_cursor

    # ------------------------------------------------------------------
    # upsert
    # ------------------------------------------------------------------

    @patch("app.memory.vector_store.register_vector")
    @patch("app.memory.vector_store.psycopg")
    def test_upsert_executes_sql(self, mock_psycopg, mock_register_vector):
        """upsert() calls execute with an INSERT … ON CONFLICT statement."""
        # Reset table-ready flag so _ensure_table runs
        import app.memory.vector_store as vs
        vs._table_ready = False

        mock_conn, mock_cursor = self._make_mock_conn()
        mock_psycopg.connect.return_value = mock_conn

        from app.memory.vector_store import upsert

        upsert(
            id="mem-1",
            memory_type="transaction",
            content="spent $50 at NTUC",
            description="grocery transaction",
            embedding=[0.1] * 1536,
        )

        # Collect all SQL strings passed to execute
        all_sql_calls = [str(c.args[0]) for c in mock_cursor.execute.call_args_list]
        assert any("ON CONFLICT" in sql for sql in all_sql_calls), (
            f"Expected 'ON CONFLICT' in one of: {all_sql_calls}"
        )

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    @patch("app.memory.vector_store.register_vector")
    @patch("app.memory.vector_store.psycopg")
    def test_search_returns_rows(self, mock_psycopg, mock_register_vector):
        """search() returns list[dict] with a 'content' key when rows exist."""
        import app.memory.vector_store as vs
        vs._table_ready = False

        fake_rows = [
            {"id": "mem-1", "memory_type": "transaction", "content": "grocery", "description": "food", "similarity": 0.9},
        ]
        mock_conn, mock_cursor = self._make_mock_conn(fetchall_return=fake_rows)
        mock_psycopg.connect.return_value = mock_conn

        from app.memory.vector_store import search

        result = search(query_embedding=[0.1] * 1536)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "content" in result[0]

    @patch("app.memory.vector_store.register_vector")
    @patch("app.memory.vector_store.psycopg")
    def test_search_filters_by_threshold(self, mock_psycopg, mock_register_vector):
        """search() passes similarity_threshold into the SQL query parameters."""
        import app.memory.vector_store as vs
        vs._table_ready = False

        mock_conn, mock_cursor = self._make_mock_conn(fetchall_return=[])
        mock_psycopg.connect.return_value = mock_conn

        from app.memory.vector_store import search

        custom_threshold = 0.75
        search(query_embedding=[0.1] * 1536, similarity_threshold=custom_threshold)

        # Verify the threshold was passed as an execute parameter
        all_execute_calls = mock_cursor.execute.call_args_list
        threshold_found = False
        for c in all_execute_calls:
            args = c.args
            # args[1] are the query params tuple
            if len(args) > 1 and custom_threshold in args[1]:
                threshold_found = True
                break
        assert threshold_found, (
            f"Expected threshold {custom_threshold} in execute params. Calls: {all_execute_calls}"
        )

    @patch("app.memory.vector_store.register_vector")
    @patch("app.memory.vector_store.psycopg")
    def test_search_empty_when_no_rows(self, mock_psycopg, mock_register_vector):
        """search() returns [] when the database returns no rows."""
        import app.memory.vector_store as vs
        vs._table_ready = False

        mock_conn, mock_cursor = self._make_mock_conn(fetchall_return=[])
        mock_psycopg.connect.return_value = mock_conn

        from app.memory.vector_store import search

        result = search(query_embedding=[0.1] * 1536)

        assert result == []

    @patch("app.memory.vector_store.register_vector")
    @patch("app.memory.vector_store.psycopg")
    def test_pg_failure_in_search_propagates(self, mock_psycopg, mock_register_vector):
        """search() propagates psycopg connection errors to the caller."""
        import app.memory.vector_store as vs
        vs._table_ready = False

        mock_psycopg.connect.side_effect = Exception("connection refused")

        from app.memory.vector_store import search

        with pytest.raises(Exception, match="connection refused"):
            search(query_embedding=[0.1] * 1536)

    # ------------------------------------------------------------------
    # _ensure_table idempotency
    # ------------------------------------------------------------------

    @patch("app.memory.vector_store.register_vector")
    @patch("app.memory.vector_store.psycopg")
    def test_ensure_table_idempotent(self, mock_psycopg, mock_register_vector):
        """_ensure_table() only runs CREATE TABLE SQL once across repeated calls."""
        import app.memory.vector_store as vs
        vs._table_ready = False  # reset so the first call actually runs

        mock_conn, mock_cursor = self._make_mock_conn()
        mock_psycopg.connect.return_value = mock_conn

        from app.memory.vector_store import _ensure_table

        _ensure_table()
        first_call_count = mock_cursor.execute.call_count

        # Second call — table flag is True, so no SQL should be issued
        _ensure_table()
        second_call_count = mock_cursor.execute.call_count

        assert second_call_count == first_call_count, (
            "CREATE TABLE SQL was executed more than once; _ensure_table is not idempotent"
        )
