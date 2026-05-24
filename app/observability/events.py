"""
Structured trace event logger — the single entry point for all observability events.

Every notable occurrence (supervisor decision, tool call, state snapshot, …) is
recorded via ``log_trace_event()``, which **simultaneously** writes to:

1. Python's ``logging`` framework (console — for real-time viewing via
   ``JsonLogFormatter`` when ``SMARTFIN_LOG_FORMAT=json``).
2. A ``.jsonl`` trace file under ``.smartfin/traces/{yyyymmdd}/{trace_id}.jsonl``
   (persistent storage — for post-hoc analysis with ``jq`` or any JSON-lines
   consumer).

Event types (constants)::

    SUPERVISOR_DECISION — routing decision with full context
    TRACE_STEP          — node execution boundary
    TOOL_CALL           — ReAct tool invocation
    STATE_SNAPSHOT      — AppState at a decision point
    TOKEN_USAGE         — LLM token consumption
    ERROR_CATEGORISED   — classified error
    API_REQUEST         — HTTP request/response

Error category constants::

    LLM_ERROR, TOOL_ERROR, STATE_ERROR, VALIDATION_ERROR, INTERNAL_ERROR

Usage::

    from app.observability.events import log_trace_event

    log_trace_event(
        "supervisor_decision",
        intent="budget_planning",
        active_agent="expense_analysis",
        pending_intent="budget_planning",
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

from app.observability.tracer import get_current_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

SUPERVISOR_DECISION = "supervisor_decision"
TRACE_STEP = "trace_step"
TOOL_CALL = "tool_call"
STATE_SNAPSHOT = "state_snapshot"
TOKEN_USAGE = "token_usage"
ERROR_CATEGORISED = "error_categorised"
API_REQUEST = "api_request"

# ---------------------------------------------------------------------------
# Error category constants
# ---------------------------------------------------------------------------

LLM_ERROR = "LLM_ERROR"
TOOL_ERROR = "TOOL_ERROR"
STATE_ERROR = "STATE_ERROR"
VALIDATION_ERROR = "VALIDATION_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"

# ---------------------------------------------------------------------------
# JSONL file writer — append-only per thread (session)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACES_ROOT = _REPO_ROOT / ".smartfin" / "traces"


def _ensure_trace_dir() -> Path:
    """Return the date-partitioned directory for today's traces."""
    day = date.today().isoformat().replace("-", "")  # e.g. 20260523
    dirpath = _TRACES_ROOT / day
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def _jsonl_path(session_id: str) -> Path:
    """File per session (thread_id).  Falls back to ``_unbound`` when no ID."""
    return _ensure_trace_dir() / f"{session_id or '_unbound'}.jsonl"


# ---------------------------------------------------------------------------
# Structured event helper
# ---------------------------------------------------------------------------

# Fields automatically injected from TraceContext
_TRACE_EXTRA_FIELDS = frozenset({
    "event", "trace_id", "span_id", "parent_span_id",
    "thread_id", "node",
    # Category extra fields used by JsonLogFormatter
    "agent", "model", "guardrail",
})


# Fields reserved by Python's LogRecord — must not appear in ``extra``
_LOG_RECORD_RESERVED = frozenset({
    "message", "args", "name", "levelname", "levelno", "pathname",
    "filename", "module", "lineno", "funcName", "created", "asctime",
    "msecs", "relativeCreated", "thread", "threadName", "process",
    "processName", "exc_info", "exc_text", "stack_info",
})


def _build_event_dict(
    event_type: str, *, node: str | None = None, **extra: Any,
) -> dict[str, Any]:
    """Assemble a flat dict with trace context + event-specific fields."""
    ctx = get_current_context()

    payload: dict[str, Any] = {
        "event": event_type,
        "node": node or ctx.node_name or "",
    }

    # Trace context
    if ctx.trace_id:
        payload["trace_id"] = ctx.trace_id
    if ctx.thread_id:
        payload["thread_id"] = ctx.thread_id
    if ctx.span_id:
        payload["span_id"] = ctx.span_id
        payload["parent_span_id"] = ctx.parent_span_id

    payload.update(extra)
    return payload


def log_trace_event(
    event_type: str, *, node: str | None = None, **extra: Any,
) -> None:
    """Emit a structured trace event.

    The event is written through two channels simultaneously:

    * **Logger** — uses ``logger.info(extra={...})`` so the existing
      ``JsonLogFormatter`` serialises it (or the plain-text formatter
      for interactive use).
    * **JSONL file** — appended to ``.smartfin/traces/{yyyymmdd}/{trace_id}.jsonl``
      for durable storage and offline querying.

    Args:
        event_type: One of the ``SUPERVISOR_DECISION``, ``TOOL_CALL``, … constants.
        node:       Override for the current span's node name.
        **extra:    Event-specific fields (intent, tool_name, duration_ms, …).
    """
    payload = _build_event_dict(event_type, node=node, **extra)

    # --- Channel 1: logging framework (console) ---
    # Strip reserved LogRecord keys from the extra dict so Python logging
    # doesn't raise "Attempt to overwrite 'message' in LogRecord".
    logger_extra = {
        k: v for k, v in payload.items()
        if k not in _LOG_RECORD_RESERVED
    }
    _text_summary = _compact_summary(payload)
    logger.info(
        "[obs] %s | %s%s",
        event_type,
        payload.get("node", ""),
        _text_summary,
        extra=logger_extra,
    )

    # --- Channel 2: JSONL file (persistent, append per session) ---
    session_id = payload.get("thread_id") or payload.get("trace_id") or ""
    if session_id:
        try:
            line = json.dumps(payload, ensure_ascii=False, default=str)
            path = _jsonl_path(session_id)
            # Append — one file per session, events in chronological order
            with open(str(path), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            logger.warning("[obs] Failed to write trace file for %s", session_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compact_summary(payload: dict[str, Any]) -> str:
    """Build a short text suffix for the plain-text log handler."""
    parts: list[str] = []
    for key in (
        "intent", "active_agent", "pending_intent",
        "tool_name", "error_category", "duration_ms",
        "status_code", "action", "confirmed",
    ):
        val = payload.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    # Common compound fields
    cats = payload.get("categorised_count")
    if cats is not None:
        parts.append(f"cats={cats}")
    txns = payload.get("transactions_count")
    if txns is not None:
        parts.append(f"txns={txns}")
    return f" [{', '.join(parts)}]" if parts else ""
