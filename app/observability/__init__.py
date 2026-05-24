"""
SmartFin Observability — structured trace event logging.

Provides a lightweight, context-aware observability layer built on Python's
standard library.  Every HTTP request gets a unique ``trace_id`` that flows
through the LangGraph node execution chain via ``contextvars``, and each
notable event is written to both the console logger and a persistent
``.jsonl`` trace file on disk.

Usage (high-level flow):

    1. HTTP request arrives → ``api.py`` middleware calls ``init_trace()``
    2. Each LangGraph node → ``with enter_span("node_name"):``
    3. At decision points → ``log_trace_event(event_type="...", **fields)``
    4. On completion → events are queryable from ``.smartfin/traces/``

Public API:
    - ``init_trace``, ``enter_span``, ``get_current_context`` from ``tracer``
    - ``log_trace_event``, event type/error constants from ``events``
"""

from app.observability.tracer import init_trace, enter_span, get_current_context, set_thread_id
from app.observability.events import log_trace_event

__all__ = [
    "init_trace",
    "enter_span",
    "get_current_context",
    "set_thread_id",
    "log_trace_event",
]
