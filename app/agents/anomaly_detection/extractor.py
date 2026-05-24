"""
Anomaly detection extractor — LLM-based explanation layer.

Responsible for:
1. Calling local statistical anomaly detection
2. Generating natural-language explanations for each flagged transaction
3. Formatting and returning results to the user

The LLM does NOT decide whether something is anomalous — it only explains
the statistical findings in natural language.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from app.agents.anomaly_detection.detector import detect_anomalies
from app.config import resolve_model_name, get_prompt
from app.state import AnomalyFlag, Transaction

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds


# ---------------------------------------------------------------------------
# LLM structured-output schema
# ---------------------------------------------------------------------------


class _FlagExplanation(BaseModel):
    transaction_id: str
    explanation: str


class _ExplanationBatch(BaseModel):
    results: list[_FlagExplanation]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_and_detect(
    messages: list,
    transactions: list[Transaction],
) -> tuple[list[AnomalyFlag], str]:
    """
    Run anomaly detection and generate natural-language explanations.

    Step 1: Call local statistical anomaly detection.
    Step 2: If flags exist, ask LLM to generate natural-language explanations.
    Step 3: Format and return both flags and explanation text.

    Args:
        messages: LangGraph message list (for understanding user context)
        transactions: List of transactions to check

    Returns:
        (flags, explanation_text)
        flags: List of AnomalyFlag objects with statistical explanations
        explanation_text: Formatted string with LLM-generated natural language
    """
    # Step 1: Run statistical detection
    flags = detect_anomalies(transactions)

    if not flags:
        return [], "No anomalous transactions detected."

    # Step 2: Generate LLM explanations for each flag
    flag_explanations = _generate_explanations(flags, transactions)

    # Step 3: Format and return
    explanation_text = _format_output(flags, flag_explanations, transactions)
    return flags, explanation_text


def _generate_explanations(
    flags: list[AnomalyFlag],
    transactions: list[Transaction],
) -> dict[str, str]:
    """
    Query LLM to generate natural-language explanations for each flagged transaction.

    Returns a dict mapping transaction_id → natural language explanation.
    On LLM failure, returns a dict using statistical explanations as fallback.
    """
    # Build a lookup for transaction details
    txn_map = {t.id: t for t in transactions}

    # Build prompt: include user context + flagged transactions + reasons
    lines = []
    for flag in flags:
        t = txn_map.get(flag.transaction_id)
        if t:
            lines.append(
                f"Transaction ID: {t.id}\n"
                f"Merchant: {t.merchant}\n"
                f"Amount: £{t.amount:.2f}\n"
                f"Category: {t.category}\n"
                f"Date: {t.date.strftime('%Y-%m-%d')}\n"
                f"Statistical reason: {flag.explanation}\n"
            )

    messages = get_prompt("anomaly_explainer").format_messages(
        flagged_transactions_text="---\n".join(lines)
    )

    model_name = resolve_model_name(os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5"))

    try:
        llm = ChatAnthropic(model=model_name, timeout=_LLM_TIMEOUT)
        structured_llm = llm.with_structured_output(_ExplanationBatch)
    except Exception as exc:
        logger.warning("Failed to initialise LLM for anomaly explanation: %s", exc)
        return {f.transaction_id: f.explanation for f in flags}

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response: _ExplanationBatch = structured_llm.invoke(messages)
            return {r.transaction_id: r.explanation for r in response.results}
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "LLM explanation failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM explanation failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    return {f.transaction_id: f.explanation for f in flags}


def _format_output(
    flags: list[AnomalyFlag],
    explanations: dict[str, str],
    transactions: list[Transaction],
) -> str:
    """
    Format the final output string.
    """
    if not flags:
        return "No anomalous transactions detected."

    txn_map = {t.id: t for t in transactions}
    lines = ["The following transactions may be anomalous:"]

    for idx, flag in enumerate(flags, start=1):
        t = txn_map.get(flag.transaction_id)
        if t:
            explanation = explanations.get(flag.transaction_id, flag.explanation)
            date_str = t.date.strftime("%Y-%m-%d") if isinstance(t.date, datetime) else str(t.date)
            lines.append(
                f"{idx}. {t.merchant} £{t.amount:.2f} on {date_str} — {explanation}"
            )

    return "\n".join(lines)
