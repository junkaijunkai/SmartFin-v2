"""
SmartFin LiteLLM gateway guardrails.

Loaded by the LiteLLM proxy via gateway/config.yaml (guardrails section).
- async_pre_call_hook:          blocks prompt injection attempts before the LLM call
- async_post_call_success_hook: redacts sensitive patterns in LLM output
"""
from __future__ import annotations

import re
import logging
from typing import Any

from litellm.integrations.custom_guardrail import CustomGuardrail  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# --- copied from app/guardrails/input_filter.py ---
INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above|earlier) instructions",
    r"disregard (all|the|previous|earlier) instructions",
    r"you are now",
    r"system prompt",
    r"developer message",
    r"reveal .* prompt",
    r"show .* hidden",
    r"bypass (the )?(guardrails|safety)?",
    r"jailbreak",
    r"act as .* instead",
    r"print .*api key",
    r"reveal .*api key",
    r"show .*token",
    r"dump .*secret",
    r"return .*credential",
]

# --- copied from app/guardrails/output_validator.py ---
_OUTPUT_PATTERNS: dict[str, str] = {
    "anthropic_key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "generic_api_key": r"(?i)api[_ -]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}",
    "bearer_token": r"(?i)bearer\s+[A-Za-z0-9._-]{12,}",
    "credit_card": r"\b(?:\d[ -]*?){13,19}\b",
}


def _normalize(text: str) -> str:
    """Collapse whitespace for pattern matching."""
    return " ".join(text.lower().split())


def _extract_last_user_message(messages: list[dict]) -> str:
    """Return the most recent user-role message content, or empty string."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


def _get_response_text(response: Any) -> str:
    """Extract text content from a LiteLLM response object."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _set_response_text(response: Any, text: str) -> None:
    """Write sanitized text back into the response object."""
    try:
        response.choices[0].message.content = text
    except (AttributeError, IndexError, TypeError):
        pass


class SmartFinGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # 框架 await，所以签名必须 async
    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ):
        """Block requests containing prompt injection patterns."""
        messages = data.get("messages", [])
        last_user = _extract_last_user_message(messages)
        if not last_user:
            return

        normalized = _normalize(last_user)
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, normalized):
                logger.warning(
                    "[guardrail] injection pattern matched, blocking request. pattern=%r",
                    pattern,
                )
                raise ValueError(f"guardrail_block: injection pattern matched")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict,
        response: Any,
    ):
        """Redact sensitive content patterns from LLM output."""
        text = _get_response_text(response)
        if not text:
            return

        redacted = text
        for name, pattern in _OUTPUT_PATTERNS.items():
            new_text = re.sub(pattern, "[REDACTED]", redacted)
            if new_text != redacted:
                logger.info("[guardrail] redacted pattern in output: %s", name)
                redacted = new_text

        if redacted != text:
            _set_response_text(response, redacted)
