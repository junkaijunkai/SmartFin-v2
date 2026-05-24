from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict


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

@dataclass(frozen=True)
class InputGuardrailResult:
    allowed: bool
    sanitized_text: str
    matched_rules: tuple[str, ...]
    reason: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _contains_injection_attempt(message: str) -> list[str]:
    hits: list[str] = []
    lower_msg = message.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_msg):
            hits.append(pattern)

    return hits


def scan_input(text: str) -> InputGuardrailResult:
    normalized = _normalize(text)
    matched_rules = _contains_injection_attempt(normalized)

    if matched_rules:
        return InputGuardrailResult(
            allowed=False,
            sanitized_text=normalized,
            matched_rules=tuple(matched_rules),
            reason="Potential prompt injection or sensitive instruction detected.",
        )

    return InputGuardrailResult(
        allowed=True,
        sanitized_text=normalized,
        matched_rules=(),
        reason="",
    )


def filter_user_input(message: str) -> Dict[str, Any]:
    """
    Lightweight input guardrail for SmartFin.

    Returns:
    {
        "allowed": bool,
        "sanitized_message": str,
        "risk_level": "low" | "medium" | "high",
        "reasons": list[str],
    }
    """
    stripped = _normalize(message)

    reasons: list[str] = []
    risk_level = "low"
    allowed = True

    if not stripped:
        return {
            "allowed": False,
            "sanitized_message": "",
            "risk_level": "medium",
            "reasons": ["empty_input"],
        }

    guardrail_result = scan_input(stripped)
    if not guardrail_result.allowed:
        allowed = False
        risk_level = "high"
        reasons.append("potential_prompt_injection")
        reasons.extend([f"matched:{hit}" for hit in guardrail_result.matched_rules])

    sanitized_message = stripped[:2000]

    return {
        "allowed": allowed,
        "sanitized_message": sanitized_message,
        "risk_level": risk_level,
        "reasons": reasons,
    }