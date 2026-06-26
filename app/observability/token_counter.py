"""
Token usage tracking — LangChain callback handler that captures LLM token counts.

Every ``ChatAnthropic`` response includes ``usage.input_tokens`` and
``usage.output_tokens``.  This handler intercepts ``on_llm_end`` and emits a
structured ``token_usage`` event via ``log_trace_event``.

Thread-safe: LangChain invokes callbacks on the same thread as the LLM call,
so ``contextvars`` (trace context) are visible.

Usage:
    from app.observability.token_counter import token_handler

    # Pass via graph config:
    config = {"configurable": {"thread_id": "..."}, "callbacks": [token_handler]}
    app_graph.stream(None, config)
"""

# DEPRECATED: Token tracking has moved to the LiteLLM gateway layer.
# This module is kept for reference. token_handler is no longer injected
# as a LangChain callback. See gateway/config.yaml for gateway-level observability.

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.observability.events import log_trace_event, TOKEN_USAGE

logger = logging.getLogger(__name__)


class TokenCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records LLM token consumption.

    Attach to the graph's ``config`` dict::

        graph.stream(input, {"configurable": {...}, "callbacks": [TokenCallbackHandler()]})
    """

    def __init__(self) -> None:
        super().__init__()
        # Track cumulative totals for the current request
        self.total_input: int = 0
        self.total_output: int = 0

    @property
    def always_verbose(self) -> bool:
        """Make sure we're called even when verbose=False (the default)."""
        return True

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs: Any) -> None:
        """Capture token usage from the LLM response and emit a trace event."""
        if not hasattr(response, "llm_output") or not response.llm_output:
            return

        usage = response.llm_output or {}
        input_tokens = usage.get("token_usage", {}).get("prompt_tokens", 0) or \
                       usage.get("token_usage", {}).get("input_tokens", 0) or 0
        output_tokens = usage.get("token_usage", {}).get("completion_tokens", 0) or \
                        usage.get("token_usage", {}).get("output_tokens", 0) or 0
        total_tokens = usage.get("token_usage", {}).get("total_tokens", 0) or 0

        # Fallback for Anthropic-style usage (separate dict key)
        if not total_tokens:
            total_tokens = input_tokens + output_tokens

        self.total_input += input_tokens
        self.total_output += output_tokens

        model = usage.get("model_name", "") or ""

        log_trace_event(
            TOKEN_USAGE,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
        )

    def reset(self) -> None:
        """Zero cumulative counters (call between requests to keep per-request totals)."""
        self.total_input = 0
        self.total_output = 0


# Module-level singleton so all graph nodes share the same handler instance.
token_handler = TokenCallbackHandler()
