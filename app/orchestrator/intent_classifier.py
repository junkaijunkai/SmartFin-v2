"""
Intent classification — routes user messages to appropriate agents using LLM.

Responsibilities:
  1. Parse user message and classify intent (which agent to invoke).
  2. Use get_llm() with structured output to ask Claude which agent to route to.
  3. Fall back to keyword matching if LLM fails.

Public API:
    classify_intent(message: str) -> str
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel
from app.config import get_llm, get_prompt
from app.tools.cache import get_cached_llm_response, cache_llm_response

logger = logging.getLogger(__name__)


class _IntentResult(BaseModel):
    """Structured output schema for intent classification."""

    agent: Literal[
        "expense_analysis",
        "budget_planning",
        "goal_planning",
        "anomaly_detection",
        "health_assessment",
        "unknown",
    ]
    reasoning: str


def _keyword_fallback(message: str) -> str:
    """
    Fallback classification using simple keyword matching.
    Provides deterministic routing when LLM unavailable.
    """
    msg = message.lower()
    msg_compact = msg.replace(" ", "")  # collapses intra-word spaces (e.g. "sa ve" → "save")

    if "budget" in msg or "budget" in msg_compact:
        return "budget_planning"
    elif any(kw in msg or kw in msg_compact for kw in ["goal", "save", "saving", "fund", "deposit"]):
        return "goal_planning"
    elif any(kw in msg or kw in msg_compact for kw in ["suspicious", "anomal"]):
        return "anomaly_detection"
    elif any(kw in msg or kw in msg_compact for kw in ["health", "risk"]):
        return "health_assessment"
    elif any(kw in msg or kw in msg_compact for kw in ["spend", "spending", "spent", "expense", "transaction", "categor", "analyse", "analyze"]):
        return "expense_analysis"

    return "unknown"


def classify_intent(message: str) -> str:
    """
    Classify user intent and return the agent name to route to.

    Falls back to keyword matching on any LLM error, ensuring routing
    always succeeds even if the API is unavailable.
    """
    # Check Redis cache first
    cached = get_cached_llm_response("intent_classifier", message)
    if cached is not None:
        logger.debug("[intent_classifier] cache hit → %s", cached.get("agent"))
        return cached["agent"]

    try:
        llm = get_llm("intent")
        structured_llm = llm.with_structured_output(_IntentResult)

        messages = get_prompt("intent_classifier").format_messages(message=message)
        result: _IntentResult = structured_llm.invoke(messages)

        # If the LLM returns "unknown", double-check with keyword fallback.
        # The LLM can be overly conservative with informal phrasing (e.g.
        # "save 1000 by June for a phone"), while keyword matching is more
        # reliable for common savings/finance patterns.
        if result.agent == "unknown":
            fallback = _keyword_fallback(message)
            if fallback != "unknown":
                logger.debug(
                    "[intent_classifier] LLM returned 'unknown', keyword fallback overrode → %s",
                    fallback,
                )
                cache_llm_response("intent_classifier", message, {"agent": fallback})
                return fallback

        logger.debug(
            "[intent_classifier] classified '%s' → %s (reasoning: %s)",
            message[:50],
            result.agent,
            result.reasoning,
        )
        cache_llm_response("intent_classifier", message, {"agent": result.agent})
        return result.agent

    except Exception as exc:
        # Gateway returns 400 when a guardrail blocks the request.
        # Distinguish security blocks from transient LLM errors.
        try:
            import openai
            if isinstance(exc, openai.BadRequestError) and "guardrail_block" in str(exc):
                logger.warning("[intent_classifier] gateway blocked request (guardrail)")
                return "blocked"
        except ImportError:
            pass
        logger.warning(
            "[intent_classifier] LLM classification failed, falling back to keyword match: %s",
            exc,
        )
        fallback = _keyword_fallback(message)
        logger.debug("[intent_classifier] keyword fallback → %s", fallback)
        return fallback
