"""
Message context management — compression and history summarisation.

Goals:
  1. Prevent the ``messages`` list from growing unbounded (controls token cost).
  2. Preserve semantic context from earlier conversation turns via LLM summarisation.
  3. Only pay the summarisation cost when the conversation actually grows long.

Trigger (``should_compress``):
  - Messages > MAX_COUNT (30)  AND
  - Estimated tokens > MAX_EST_TOKENS (60 000)

Output after compression:
  [AIMessage(<llm_summary>)] + [last N complete messages]
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_COUNT = 30          # message count threshold
_MAX_EST_TOKENS = 60000  # rough token estimate threshold
_KEEP_RECENT = 10        # how many recent messages to preserve verbatim
_CHARS_PER_TOKEN = 4     # rough estimate (English-heavy text)

# ---------------------------------------------------------------------------
# Threshold check
# ---------------------------------------------------------------------------


def should_compress(messages: list) -> bool:
    """Return True if the message list exceeds size or token thresholds."""
    if len(messages) <= _MAX_COUNT:
        return False
    rough_tokens = sum(
        len(str(getattr(m, "content", "") or "")) // _CHARS_PER_TOKEN
        for m in messages
    )
    return rough_tokens > _MAX_EST_TOKENS


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def compress_messages(
    messages: list,
    *,
    agent_name: str | None = None,
) -> list:
    """
    Compress ``messages`` if thresholds are exceeded.

    Strategy:
      1. Run ``should_compress`` — if False, return messages unchanged.
      2. Split into ``old`` (all but the last N) and ``recent`` (last N).
      3. Call LLM to summarise ``old`` into a single AIMessage.
      4. Return ``[summary] + recent``.

    The ``agent_name`` parameter is used only for logging; it does not
    affect behaviour.
    """
    if not should_compress(messages):
        return messages

    old = messages[:-_KEEP_RECENT]
    recent = messages[-_KEEP_RECENT:]

    summary = _build_summary_with_llm(old)

    hint = (
        f"[Conversation history compressed: {len(old)} earlier messages "
        f"summarised below; the last {len(recent)} messages are shown verbatim.]"
    )
    logger.info(
        "compress_messages%s: %d -> 1 + %d",
        f" [{agent_name}]" if agent_name else "",
        len(messages),
        len(recent),
    )

    return [AIMessage(content=hint), summary] + recent


# ---------------------------------------------------------------------------
# LLM summary
# ---------------------------------------------------------------------------


def _build_summary_with_llm(old_messages: list) -> AIMessage:
    """
    Use Claude to summarise older messages into a concise history summary.

    Falls back to a text-concatenation approach if the LLM is unavailable.
    """
    from app.config import get_llm, get_react_prompt

    # Build a compact, lossy-but-helpful text representation
    lines: list[str] = []
    for msg in old_messages:
        role = (
            "User"
            if isinstance(msg, HumanMessage)
            or getattr(msg, "type", "") == "human"
            else "Assistant"
        )
        content = str(getattr(msg, "content", "") or "")
        # Truncate very long individual messages for the summariser prompt
        if len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"{role}: {content}")

    transcript = "\n".join(lines)

    system = get_react_prompt(
        "summarise_history",
        transcript=transcript,
        old_count=len(old_messages),
    )

    try:
        llm = get_llm(timeout=30)
        response = llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Summarise the key points from this conversation "
                        "transcript. Focus on:\n"
                        "  1. Financial goals the user has set\n"
                        "  2. Budget decisions made\n"
                        "  3. Any data the user provided (income, transactions)\n"
                        "  4. Key findings from agents\n\n"
                        f"Transcript:\n{transcript}"
                    )
                )
            ]
        )
        summary = response.content if response.content else ""
    except Exception as exc:
        logger.warning(
            "[context] LLM summarisation failed, using fallback: %s", exc
        )
        summary = _fallback_summary(old_messages)

    return AIMessage(content=summary)


def _fallback_summary(old_messages: list) -> str:
    """Build a simple text-based summary when LLM is unavailable."""
    turn_count = len(old_messages) // 2
    return (
        f"[Earlier conversation: ~{turn_count} user-assistant exchanges. "
        f"The user has been discussing their finances with SmartFin. "
        f"All relevant structured data is in the shared state fields.]"
    )
