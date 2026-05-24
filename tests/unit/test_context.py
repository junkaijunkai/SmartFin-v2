"""Unit tests for message context management — compression and summarisation."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.orchestrator.context import (
    should_compress,
    compress_messages,
    _fallback_summary,
    _build_summary_with_llm,
    _MAX_COUNT,
    _MAX_EST_TOKENS,
    _KEEP_RECENT,
)


# ---------------------------------------------------------------------------
# should_compress
# ---------------------------------------------------------------------------


def make_msg(content: str, role: str = "human") -> HumanMessage | AIMessage:
    cls = HumanMessage if role == "human" else AIMessage
    return cls(content=content)


class TestShouldCompress:
    def test_below_message_count_returns_false(self):
        msgs = [make_msg("hello")] * (_MAX_COUNT - 1)
        assert should_compress(msgs) is False

    def test_exactly_at_count_returns_false(self):
        msgs = [make_msg("hello")] * _MAX_COUNT
        assert should_compress(msgs) is False

    def test_above_count_but_low_tokens_returns_false(self):
        msgs = [make_msg("a")] * (_MAX_COUNT + 1)
        assert should_compress(msgs) is False

    def test_above_count_and_high_tokens_returns_true(self):
        # Each message is ~8000 chars → ~2000 tokens.  31 msgs → ~62000 tokens.
        long_content = "x" * 8000
        msgs = [make_msg(long_content)] * (_MAX_COUNT + 1)
        assert should_compress(msgs) is True

    def test_empty_list_returns_false(self):
        assert should_compress([]) is False

    def test_non_string_content_uses_str_fallback(self):
        # AIMessage wraps non-string content; str() on content works via fallback
        msgs = [AIMessage(content=b"x" * 8000)] * (_MAX_COUNT + 1)
        assert should_compress(msgs) is True

    def test_boundary_just_below_token_threshold(self):
        # Total chars / 4 slightly below _MAX_EST_TOKENS
        total_chars = _MAX_EST_TOKENS * 4 - 1
        per_msg = total_chars // (_MAX_COUNT + 1)
        msgs = [make_msg("x" * per_msg)] * (_MAX_COUNT + 1)
        assert should_compress(msgs) is False


# ---------------------------------------------------------------------------
# compress_messages
# ---------------------------------------------------------------------------


class TestCompressMessages:
    def test_below_threshold_returns_same_list(self):
        msgs = [make_msg("hello")] * 5
        result = compress_messages(msgs)
        assert result is msgs  # same object, no copy

    def test_compresses_when_above_threshold(self):
        long = "x" * 8000  # ~2000 tokens each; 35 msgs → ~70000 tokens
        msgs = [make_msg(long)] * (_MAX_COUNT + 5)

        with patch(
            "app.orchestrator.context._build_summary_with_llm"
        ) as mock_summary:
            mock_summary.return_value = AIMessage(content="summary text")
            result = compress_messages(msgs)

        # Should be: [hint, summary] + recent N
        assert len(result) == _KEEP_RECENT + 2
        assert result[0].content.startswith("[Conversation history compressed")
        assert result[1].content == "summary text"
        # Last _KEEP_RECENT messages should be the recent original ones
        assert result[-1].content == long

    def test_logs_agent_name_when_provided(self):
        long = "x" * 8000
        msgs = [make_msg(long)] * (_MAX_COUNT + 5)

        with patch(
            "app.orchestrator.context._build_summary_with_llm"
        ) as mock_summary:
            mock_summary.return_value = AIMessage(content="s")
            with patch(
                "app.orchestrator.context.logger"
            ) as mock_logger:
                compress_messages(msgs, agent_name="test_agent")

        mock_logger.info.assert_called_once()
        args, _ = mock_logger.info.call_args
        # args[0] is the format string, args[1] is the agent_name suffix
        combined = "".join(str(a) for a in args)
        assert "test_agent" in combined

    def test_recent_messages_preserved_in_order(self):
        """Ensure the last _KEEP_RECENT messages are in their original order."""
        long = "x" * 8000
        recent_msgs = [make_msg(f"recent-{i}" + "x" * 8000) for i in range(_KEEP_RECENT)]
        msgs = [make_msg(long)] * (_MAX_COUNT - _KEEP_RECENT + 1) + recent_msgs

        with patch(
            "app.orchestrator.context._build_summary_with_llm"
        ) as mock_summary:
            mock_summary.return_value = AIMessage(content="s")
            result = compress_messages(msgs)

        for i, expected in enumerate(recent_msgs):
            assert result[i + 2].content == f"recent-{i}" + "x" * 8000


# ---------------------------------------------------------------------------
# _build_summary_with_llm
# ---------------------------------------------------------------------------


class TestBuildSummaryWithLLM:
    # _build_summary_with_llm does `from langchain_anthropic import ChatAnthropic`
    # so we patch at the original module.
    PATCH_PATH = "langchain_anthropic.ChatAnthropic"

    def test_llm_success_returns_aimessage(self):
        old = [make_msg("hello"), make_msg("world", role="assistant")]

        with patch(self.PATCH_PATH) as MockLLM:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Concise summary here."
            mock_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_instance

            result = _build_summary_with_llm(old)

        assert isinstance(result, AIMessage)
        assert result.content == "Concise summary here."

    def test_llm_failure_falls_back(self):
        old = [make_msg("hello"), make_msg("world", role="assistant")]

        with patch(self.PATCH_PATH) as MockLLM:
            MockLLM.side_effect = RuntimeError("API unavailable")
            result = _build_summary_with_llm(old)

        assert isinstance(result, AIMessage)
        assert "Earlier conversation" in result.content
        assert "~1" in result.content  # ~1 turn (2 messages)

    def test_truncates_long_messages_in_transcript(self):
        old = [make_msg("x" * 1000)]

        with patch(self.PATCH_PATH) as MockLLM:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "summary"
            mock_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_instance

            _build_summary_with_llm(old)

        # invoke was called with [HumanMessage(...)]
        call_args = mock_instance.invoke.call_args[0]
        msgs_list = call_args[0]  # the first positional arg is the messages list
        user_msg = msgs_list[0]   # the HumanMessage
        assert "…" in str(user_msg.content)

    def test_empty_old_messages(self):
        with patch(self.PATCH_PATH) as MockLLM:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = ""
            mock_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_instance

            result = _build_summary_with_llm([])
        assert isinstance(result, AIMessage)


# ---------------------------------------------------------------------------
# _fallback_summary
# ---------------------------------------------------------------------------


class TestFallbackSummary:
    def test_generates_turn_count(self):
        result = _fallback_summary([make_msg("a"), make_msg("b", "assistant"),
                                     make_msg("c"), make_msg("d", "assistant")])
        assert "~2" in result  # 4 messages = 2 turns

    def test_empty_input(self):
        result = _fallback_summary([])
        assert "~0" in result

    def test_odd_message_count_floor_division(self):
        result = _fallback_summary([make_msg("a")])
        assert "~0" in result  # 1 // 2 = 0
