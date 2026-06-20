"""Unit tests for app/memory/retriever.py — TDD Red→Green."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from app.memory.retriever import retrieve_memory


_FAKE_EMBEDDING = [0.1] * 1536

_FAKE_SEARCH_ROW = {
    "id": "goal_planning/emergency-fund",
    "memory_type": "goal_planning",
    "content": "some content",
    "description": "desc",
    "similarity": 0.9,
}


class TestRetrieveMemory:
    def test_returns_name_content_dicts(self):
        """Happy path: embed + search succeed → returns [{name, content}]."""
        with patch("app.memory.retriever.embed", return_value=_FAKE_EMBEDDING) as mock_embed, \
             patch("app.memory.retriever.search", return_value=[_FAKE_SEARCH_ROW]) as mock_search:
            result = retrieve_memory("how is my goal?")

        assert result == [{"name": "goal_planning/emergency-fund", "content": "some content"}]

    def test_returns_empty_on_embed_failure(self):
        """embed raises an exception → returns [] without re-raising."""
        with patch("app.memory.retriever.embed", side_effect=RuntimeError("embed failed")):
            result = retrieve_memory("test message")

        assert result == []

    def test_returns_empty_on_search_failure(self):
        """search raises an exception → returns [] without re-raising."""
        with patch("app.memory.retriever.embed", return_value=_FAKE_EMBEDDING), \
             patch("app.memory.retriever.search", side_effect=Exception("DB down")):
            result = retrieve_memory("test message")

        assert result == []

    def test_returns_empty_when_no_results(self):
        """search returns [] → retrieve_memory returns []."""
        with patch("app.memory.retriever.embed", return_value=_FAKE_EMBEDDING), \
             patch("app.memory.retriever.search", return_value=[]):
            result = retrieve_memory("test message")

        assert result == []

    def test_passes_message_to_embed(self):
        """embed is called with the exact user_message string."""
        with patch("app.memory.retriever.embed", return_value=_FAKE_EMBEDDING) as mock_embed, \
             patch("app.memory.retriever.search", return_value=[]):
            retrieve_memory("track my savings")

        mock_embed.assert_called_once_with("track my savings")

    def test_passes_embedding_to_search(self):
        """search is called with the embedding returned by embed as its first arg."""
        with patch("app.memory.retriever.embed", return_value=_FAKE_EMBEDDING) as mock_embed, \
             patch("app.memory.retriever.search", return_value=[]) as mock_search:
            retrieve_memory("any message")

        args, _ = mock_search.call_args
        assert args[0] == _FAKE_EMBEDDING
