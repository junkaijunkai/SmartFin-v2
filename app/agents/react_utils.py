"""
Shared ReAct loop engine for SmartFin agents.

Provides a generic ReAct (Reasoning + Acting) loop that any agent node can use.
The loop lets the LLM iteratively reason, call tools, observe results, and
produce a final response — conforming to Anthropic's standard tool_use flow.

The loop terminates when the LLM returns a response with no tool calls
(stop_reason: end_turn). The last AIMessage's ``content`` is the natural-language
answer to show the user.

Usage inside an agent node:

    from app.agents.react_utils import run_react_loop
    from app.config import get_llm

    def my_agent_node(state: AppState) -> dict:
        tool_ctx: dict = {}

        @tool
        def my_tool(param: str) -> str:
            "Tool docstring."
            tool_ctx["key"] = result
            return str(result)

        llm = get_llm("alias").bind_tools([my_tool])
        system = get_prompt("react_my_agent").format(...)

        response, _ = run_react_loop(
            llm=llm,
            tools={"my_tool": my_tool},
            system_prompt=system,
            user_message=last_user_msg,
            tool_ctx=tool_ctx,
        )

        return {
            "field": tool_ctx.get("key"),
            "messages": [AIMessage(content=response.content)],
        }
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

from app.observability.events import log_trace_event, TOOL_CALL, ERROR_CATEGORISED, TOOL_ERROR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOOL_RESULT_CHARS = 8000
_MAX_REACT_STEPS = 15
_MAX_ARGS_LOG_CHARS = 300


def _safe_summarize(args: dict, max_len: int = _MAX_ARGS_LOG_CHARS) -> str:
    """JSON-serialize tool args, truncated to avoid bloating trace events."""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def run_react_loop(
    llm,
    tools: dict[str, Callable],
    system_prompt: str,
    user_message: str = "",
    tool_ctx: dict | None = None,
    history: list | None = None,
    max_steps: int = _MAX_REACT_STEPS,
) -> tuple[AIMessage, list[dict[str, Any]]]:
    """
    Run a ReAct (Reasoning + Acting) loop aligned with Anthropic's tool_use standard.

    The LLM receives the system prompt, optional conversation history, and the
    latest user message. It iteratively reasons, calls tools, and receives
    tool_result messages until it produces a response with no tool calls
    (stop_reason: end_turn). That final response's ``content`` is the answer.

    Parameters
    ----------
    llm :
        An LLM instance (from get_llm()) with ``.bind_tools(tool_list)`` already called.
    tools :
        Mapping from tool name to callable. The loop executes each tool call
        and appends the result as a ToolMessage before the next LLM invocation.
    system_prompt :
        System message text describing the agent's role and available tools.
    user_message :
        The user's latest input text (appended after history).
    tool_ctx :
        Optional mutable dict for tools to share execution results with
        post-processing logic (e.g. Python objects created during tool execution).
    history :
        Optional list of prior-turn messages (HumanMessage / AIMessage) to
        prepend after the system prompt. Enables multi-turn continuations where
        the LLM needs context from previous exchanges (e.g. a clarification
        follow-up to a goal creation request).
    max_steps :
        Maximum LLM-invoke iterations before force-termination.

    Returns
    -------
    (final_ai_message, tool_call_history)
        final_ai_message : the last AIMessage (the end_turn response whose
            ``content`` is the natural-language answer).
        tool_call_history : list of dicts recording each tool call in order.
    """
    if tool_ctx is None:
        tool_ctx = {}

    messages: list = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(history)
    if user_message:
        messages.append(HumanMessage(content=user_message))

    call_history: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        try:
            response: AIMessage = llm.invoke(messages)
        except Exception as exc:
            # Distinguish gateway guardrail blocks from transient LLM errors,
            # mirroring the handling in intent_classifier.py.
            try:
                import openai
                if isinstance(exc, openai.BadRequestError) and "guardrail_block" in str(exc):
                    logger.warning("[react] gateway blocked request at step %d (guardrail)", step)
                    log_trace_event(
                        ERROR_CATEGORISED, error_category=TOOL_ERROR,
                        step=step, message="guardrail_block",
                    )
                    return AIMessage(
                        content="Your request was blocked by our safety guardrails. "
                                "Please submit a normal personal finance query."
                    ), call_history
            except ImportError:
                pass
            logger.warning("[react] LLM invocation failed at step %d: %s", step, exc)
            log_trace_event(
                ERROR_CATEGORISED, error_category=TOOL_ERROR,
                step=step, message=str(exc)[:200],
            )
            return AIMessage(
                content="I encountered an error while processing your request. "
                        "Please try again."
            ), call_history

        messages.append(response)

        # --- No tool calls → end_turn, return the response ---
        if not response.tool_calls:
            logger.debug("[react] end_turn at step %d", step)
            return response, call_history

        # --- Execute each tool call and feed results back ---
        for tc in response.tool_calls:
            tool_name = tc.get("name", "unknown_tool")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            logger.debug("[react] Step %d tool call: %s(%s)", step, tool_name, tool_args)

            if tool_name not in tools:
                result_text = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {list(tools.keys())}"
                )
                log_trace_event(TOOL_CALL, tool_name=tool_name,
                                success=False, error="unknown_tool", duration_ms=0.0)
            else:
                t0 = time.perf_counter()
                try:
                    # LangChain StructuredTool must be invoked via .invoke()
                    raw = tools[tool_name].invoke(tool_args)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    if raw is None:
                        result_text = "Success (no output)"
                    elif isinstance(raw, str):
                        result_text = raw
                    else:
                        try:
                            result_text = json.dumps(raw, ensure_ascii=False, default=str)
                        except (TypeError, ValueError):
                            result_text = str(raw)
                    log_trace_event(TOOL_CALL, tool_name=tool_name,
                                    args_summary=_safe_summarize(tool_args),
                                    duration_ms=round(duration_ms, 1), success=True)
                except Exception as exc:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.warning("[react] Tool '%s' failed: %s", tool_name, exc)
                    result_text = f"Error executing {tool_name}: {exc}"
                    log_trace_event(TOOL_CALL, tool_name=tool_name,
                                    args_summary=_safe_summarize(tool_args),
                                    duration_ms=round(duration_ms, 1),
                                    success=False, error=str(exc)[:200])
                    log_trace_event(ERROR_CATEGORISED, error_category=TOOL_ERROR,
                                    tool_name=tool_name, message=str(exc)[:200])

            if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                logger.warning(
                    "[react] Truncated tool '%s' result from %d to %d chars",
                    tool_name, len(result_text), _MAX_TOOL_RESULT_CHARS,
                )
                result_text = result_text[:_MAX_TOOL_RESULT_CHARS]

            messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))
            call_history.append({
                "name": tool_name,
                "args": tool_args,
                "result": result_text[:500],
            })

    # --- Max steps reached ---
    logger.warning("[react] Reached max_steps=%d without end_turn", max_steps)
    log_trace_event(ERROR_CATEGORISED, error_category=TOOL_ERROR,
                    message=f"max_steps={max_steps} reached without end_turn")
    return response, call_history
