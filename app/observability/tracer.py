"""
Trace context propagation — ``trace_id`` / ``span_id`` via ``contextvars``.

Uses Python ``contextvars`` (PEP 567) so trace metadata flows through
LangGraph's async node chain WITHOUT explicit parameter passing.

Thread-safe: ``contextvars`` are per-``asyncio.Task`` / per-thread.

Usage::

    from app.observability.tracer import init_trace, enter_span

    # At the start of an HTTP request
    init_trace(thread_id="ui-abc123")

    # Inside a LangGraph node
    with enter_span("supervisor"):
        ctx = get_current_context()
        logger.info("trace=%s span=%s", ctx.trace_id, ctx.span_id)
"""

from __future__ import annotations

import contextvars
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace context dataclass
# ---------------------------------------------------------------------------


@dataclass
class TraceContext:
    """Mutable context data attached to one HTTP request → graph execution chain."""

    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    node_name: str = ""
    thread_id: str = ""


# Thread-local / async-Task-local slot — never directly exposed outside this module.
_trace_var: contextvars.ContextVar[TraceContext] = contextvars.ContextVar(
    "smartfin_trace", default=TraceContext()
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_current_context() -> TraceContext:
    """Return the ``TraceContext`` active in the calling coroutine / thread."""
    return _trace_var.get()


def init_trace(thread_id: str = "") -> TraceContext:
    """Begin a brand-new trace — generates ``trace_id`` and resets span state.

    Typically called once at the start of an HTTP request (``api.py`` middleware).
    """
    ctx = TraceContext(
        trace_id=uuid.uuid4().hex[:12],
        span_id="",
        parent_span_id="",
        node_name="",
        thread_id=thread_id,
    )
    _trace_var.set(ctx)
    return ctx


def set_thread_id(thread_id: str) -> None:
    """Stamp the current trace with a known ``thread_id``.

    Called from ``memory_loader_node`` as soon as the thread_id is resolved.
    """
    ctx = get_current_context()
    ctx.thread_id = thread_id
    _trace_var.set(ctx)


# ---------------------------------------------------------------------------
# Span context manager
# ---------------------------------------------------------------------------


class _SpanGuard:
    """Context manager that enters/exits a child span.

    On ``__enter__`` a new ``span_id`` is generated and the previous context
    is saved.  On ``__exit__`` the parent context is restored.
    """

    def __init__(self, node_name: str) -> None:
        self._node_name = node_name
        self._parent: TraceContext | None = None

    def __enter__(self) -> TraceContext:
        parent = get_current_context()
        self._parent = parent

        child = TraceContext(
            trace_id=parent.trace_id,
            span_id=uuid.uuid4().hex[:10],
            parent_span_id=parent.span_id or parent.trace_id,
            node_name=self._node_name,
            thread_id=parent.thread_id,
        )
        _trace_var.set(child)
        logger.debug("[tracer] enter span=%s node=%s trace=%s",
                     child.span_id, self._node_name, child.trace_id)
        return child

    def __exit__(self, *args: Any) -> None:
        _trace_var.set(self._parent)
        if self._parent:
            logger.debug("[tracer] exit span=%s node=%s",
                         self._parent.span_id, self._node_name)


def enter_span(node_name: str) -> _SpanGuard:
    """Open a new child span inside the current trace.

    Example::

        with enter_span("supervisor"):
            # context now has a unique span_id under the current trace
            do_work()
    """
    return _SpanGuard(node_name)
