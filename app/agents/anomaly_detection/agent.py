"""
Anomaly Detection agent — LangGraph node entry point.

Directly runs statistical detection and builds explanations without ReAct
overhead, since the workflow is a strictly sequential two-step process
(detect → explain) with no branching decisions for the LLM to make.

Design:
  - detector.py handles IQR and frequency-based statistical detection.
  - extractor.py handles LLM-based explanation generation (with fallback).
  - This node is a thin wrapper that reads state, calls both, and formats
    the response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from app.orchestrator.state_view import AgentStateView
from app.agents.anomaly_detection.extractor import extract_and_detect

logger = logging.getLogger(__name__)


def run(view: AgentStateView) -> dict:
    """
    LangGraph node entry point for anomaly detection.

    State consumed:
      - categorised_transactions (preferred) or transactions
      - messages (for LLM explanation context)

    State produced:
      - anomaly_flags
      - anomaly_explanation
      - messages (AI response)
    """
    # ------------------------------------------------------------------
    # Read data from state
    # ------------------------------------------------------------------
    transactions = (
        view.get("categorised_transactions")
        or view.get("transactions")
        or []
    )
    messages_list = view.get("messages") or []

    if not transactions:
        return {
            "anomaly_flags": [],
            "anomaly_explanation": "No transactions available to scan.",
            "messages": [
                AIMessage(
                    content="I need transaction data to detect anomalies. "
                            "Please provide your transactions first."
                )
            ],
        }

    # ------------------------------------------------------------------
    # Run detection directly
    # ------------------------------------------------------------------
    flags, explanation = extract_and_detect(messages_list, transactions)
    summary_text = explanation or (
        f"Scanned {len(transactions)} transactions, "
        f"found {len(flags)} anomaly flag(s)."
    )

    return {
        "anomaly_flags": flags,
        "anomaly_explanation": explanation,
        "messages": [AIMessage(content=summary_text)],
    }
