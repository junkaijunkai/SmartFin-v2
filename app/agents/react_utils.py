"""
Shared ReAct loop engine for SmartFin agents.

Provides a generic ReAct (Reasoning + Acting) loop that any agent node can use.
The loop lets the LLM iteratively reason, call tools, observe results, and
produce a final answer — transforming agents from single-pass function calls
into true decision-making loops.

Usage inside an agent node:

    from app.agents.react_utils import run_react_loop
    from langchain_anthropic import ChatAnthropic

    def my_agent_node(state: AppState) -> dict:
        tool_ctx: dict = {}

        @tool
        def my_tool(param: str) -> str:
            "Tool docstring."
            tool_ctx["key"] = result
            return str(result)

        llm = ChatAnthropic(model=...).bind_tools([my_tool, final_answer])
        system = get_prompt("react_my_agent").format(...)

        final_msg, history = run_react_loop(
            llm=llm,
            tools={"my_tool": my_tool, "final_answer": final_answer},
            system_prompt=system,
            user_message=last_user_msg,
        )

        return {
            "field": tool_ctx.get("key"),
            "messages": [AIMessage(content=tool_ctx.get("summary", ""))],
        }
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

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
# Shared final_answer tool (used by every agent)
# ---------------------------------------------------------------------------


@tool
def final_answer(
    summary: str,
    needs_hitl_confirmation: bool = False,
    hitl_summary: str | None = None,
    hitl_details: list[str] | None = None,
) -> str:
    """
    Call this tool when you have completed all your analysis and are ready to
    present the final response to the user.

    Parameters
    ----------
    summary :
        The complete natural-language response to show the user. It should
        summarise what you did, what you found, and any recommendations.
    needs_hitl_confirmation :
        Whether the result requires the user to approve or reject before it is
        committed (Human-in-the-Loop pause). Set to True for actions that modify
        the user's financial data (e.g. creating a goal, committing a budget).
    hitl_summary :
        One-line summary shown in the HITL confirmation card (only when
        needs_hitl_confirmation is True).
    hitl_details :
        Bullet-point detail lines shown in the HITL card (only when
        needs_hitl_confirmation is True).
    """
    # This function is never executed directly — its output is captured by the
    # ReAct loop runner via the tool_ctx dict. The body here is a fallback
    # in case it is ever called as a plain Python function.
    return "final_answer recorded"


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def run_react_loop(
    llm,
    tools: dict[str, Callable],
    system_prompt: str,
    user_message: str = "",
    tool_ctx: dict | None = None,
    max_steps: int = _MAX_REACT_STEPS,
    final_answer_name: str = "final_answer",
) -> tuple[AIMessage, list[dict[str, Any]]]:
    """
    Run a ReAct (Reasoning + Acting) loop.

    The LLM receives the system prompt followed by the user message, then
    iteratively decides whether to call a tool or produce a final answer.
    When tools are called, their results are fed back as ToolMessage.  The
    loop terminates when the LLM produces a response with no tool calls, or
    when the ``final_answer`` tool is called (which captures structured output
    into ``tool_ctx`` and breaks immediately).

    Parameters
    ----------
    llm :
        A ChatAnthropic (or other ChatModel) instance that has already had
        ``.bind_tools(tool_list)`` called on it.
    tools :
        Mapping from tool name (as the LLM knows it) to the callable that
        implements it.  The loop looks up ``response.tool_calls[i]["name"]``
        in this dict.
    system_prompt :
        System message text — describes the agent's role and available tools.
    user_message :
        The user's latest input text.
    tool_ctx :
        Optional mutable dict.  Tools can write intermediate results here.
        When the ``final_answer`` tool is called, its structured args are
        automatically injected under the key ``"_final"``, and specific fields
        (``summary``, ``needs_hitl``, ``hitl_summary``, ``hitl_details``) are
        also written so callers can read them directly.
    max_steps :
        Maximum number of LLM-invoke iterations before we force-terminate.
    final_answer_name :
        The tool name that signals "done".  When a tool call with this name is
        encountered, its args are captured into tool_ctx and the loop breaks
        without executing the tool.

    Returns
    -------
    (final_ai_message, tool_call_history)
        final_ai_message : the last AIMessage from the LLM (the one that
            triggered the break — either a no-tool-call response, or the
            message that contained the ``final_answer`` tool call).
        tool_call_history : list of dicts in execution order.
    """
    if tool_ctx is None:
        tool_ctx = {}

    messages: list = [SystemMessage(content=system_prompt)]
    if user_message:
        messages.append(HumanMessage(content=user_message))

    history: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        try:
            response: AIMessage = llm.invoke(messages)
        except Exception as exc:
            logger.warning(
                "[react] LLM invocation failed at step %d: %s", step, exc
            )
            log_trace_event(
                ERROR_CATEGORISED, error_category=LLM_ERROR,
                step=step, message=str(exc)[:200],
            )
            error_msg = AIMessage(
                content="I encountered an error while processing your request. "
                        "Please try again."
            )
            return error_msg, history

        messages.append(response)

        # --- No tool calls → final answer ---
        if not response.tool_calls:
            logger.debug("[react] Final answer at step %d", step)
            return response, history

        # --- Execute each tool call ---
        hit_final = False
        for tc in response.tool_calls:
            tool_name = tc.get("name", "unknown_tool")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            logger.debug(
                "[react] Step %d tool call: %s(%s)", step, tool_name, tool_args
            )

            # --- final_answer is intercepted, not executed ---
            if tool_name == final_answer_name:
                log_trace_event(TOOL_CALL, tool_name="final_answer",
                                args_summary=_safe_summarize(tool_args),
                                duration_ms=0.0, success=True)
                tool_ctx["_final"] = tool_args
                tool_ctx["summary"] = tool_args.get("summary", "")
                tool_ctx["needs_hitl"] = tool_args.get(
                    "needs_hitl_confirmation", False
                )
                tool_ctx["hitl_summary"] = tool_args.get("hitl_summary")
                tool_ctx["hitl_details"] = tool_args.get("hitl_details", [])
                hit_final = True
                break

            if tool_name not in tools:
                result_text = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools: {list(tools.keys())}"
                )
                log_trace_event(TOOL_CALL, tool_name=tool_name,
                                success=False,
                                error="unknown_tool",
                                duration_ms=0.0)
            else:
                t0 = time.perf_counter()
                try:
                    # LangChain 1.3+ StructuredTool is NOT directly callable;
                    # must use .invoke() with a single arg dict.
                    raw = tools[tool_name].invoke(tool_args)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    if raw is None:
                        result_text = "Success (no output)"
                    elif isinstance(raw, str):
                        result_text = raw
                    else:
                        try:
                            result_text = json.dumps(
                                raw, ensure_ascii=False, default=str
                            )
                        except (TypeError, ValueError):
                            result_text = str(raw)
                    log_trace_event(TOOL_CALL, tool_name=tool_name,
                                    args_summary=_safe_summarize(tool_args),
                                    duration_ms=round(duration_ms, 1),
                                    success=True)
                except Exception as exc:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.warning(
                        "[react] Tool '%s' failed: %s", tool_name, exc
                    )
                    result_text = f"Error executing {tool_name}: {exc}"
                    log_trace_event(TOOL_CALL, tool_name=tool_name,
                                    args_summary=_safe_summarize(tool_args),
                                    duration_ms=round(duration_ms, 1),
                                    success=False, error=str(exc)[:200])
                    log_trace_event(ERROR_CATEGORISED,
                                    error_category=TOOL_ERROR,
                                    tool_name=tool_name,
                                    message=str(exc)[:200])

            # Truncate long results to keep context manageable
            if len(result_text) > _MAX_TOOL_RESULT_CHARS:
                logger.warning(
                    "[react] Truncated tool '%s' result from %d to %d chars",
                    tool_name,
                    len(result_text),
                    _MAX_TOOL_RESULT_CHARS,
                )
                result_text = result_text[:_MAX_TOOL_RESULT_CHARS]

            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_id)
            )
            history.append({
                "name": tool_name,
                "args": tool_args,
                "result": result_text[:500],
            })

        # --- final_answer was called → terminate loop ---
        if hit_final:
            logger.debug("[react] final_answer invoked at step %d", step)
            return response, history

    # --- Max steps reached without final answer ---
    logger.warning(
        "[react] Reached max_steps=%d without a final answer", max_steps
    )
    log_trace_event(ERROR_CATEGORISED, error_category=LLM_ERROR,
                    message=f"max_steps={max_steps} reached without final_answer")
    return response, history
