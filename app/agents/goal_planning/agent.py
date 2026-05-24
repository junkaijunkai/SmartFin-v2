"""
Goal Planning agent — LangGraph node entry point with ReAct loop.

Transformed from a single-pass function into a ReAct agent. The LLM reasons
about the user's financial goal request, extracts structured goal data,
validates it, creates financial goals, and computes required savings.

Key principle (unchanged):
  - LLM handles language understanding and extraction.
  - tracker.py remains the deterministic financial calculation tool.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from app.orchestrator.state_view import AgentStateView
from app.agents.react_utils import run_react_loop, final_answer as shared_final_answer
from app.agents.goal_planning.tracker import calculate_required_monthly_saving as _calc_saving
from app.agents.goal_planning.extractor import extract_goal_from_message, GoalExtractionResult
from app.config import resolve_model_name, get_react_prompt
from app.state import AppState, FinancialGoal

logger = logging.getLogger(__name__)


def _get_latest_message_text(view: AgentStateView) -> str:
    messages = view.get("messages") or []
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    return str(content) if content is not None else ""


def run(view: AgentStateView) -> dict:
    """
    LangGraph node entry point with ReAct loop for goal planning.

    State consumed:
      - messages (latest user message)
      - goals (existing goals)
      - monthly_income
      - budget_allocations
      - current_date

    State produced:
      - goals (updated list)
      - pending_confirmation (for HITL on new goals)
      - hitl_rollback (snapshot of original goal IDs)
    """
    # ------------------------------------------------------------------
    # Read context from state
    # ------------------------------------------------------------------
    goals = list(view.get("goals") or [])
    original_goal_ids = [g.id for g in goals]
    monthly_income = view.get("monthly_income") or 0.0
    budget_allocations = view.get("budget_allocations") or []
    total_spent = sum(
        getattr(a, "spent_amount", 0) for a in budget_allocations
    )
    monthly_surplus = monthly_income - total_spent

    latest_message = _get_latest_message_text(view)
    current_date_str = view.get("current_date")
    today = (
        date.fromisoformat(current_date_str)
        if current_date_str
        else date.today()
    )

    extractor_context: str | None = None

    if not latest_message.strip():
        # No user message — evaluate existing goals without ReAct loop
        final_goals: list[FinancialGoal] = []
        for goal in goals:
            required = _calc_saving(goal)
            on_track = required <= monthly_surplus
            final_goals.append(
                goal.model_copy(
                    update={
                        "required_monthly_saving": required,
                        "on_track": on_track,
                    }
                )
            )
        return {"goals": final_goals}

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------
    tool_ctx: dict[str, Any] = {}

    @tool
    def extract_goal_tool(
        user_message: str,
        reference_date: str,
    ) -> str:
        """
        Analyse a user's natural-language message and extract structured
        financial goal information: whether they intend to set a goal,
        the goal name, target amount, target date, and current savings.

        Provide the user's original message and today's date (YYYY-MM-DD).
        Returns structured extraction results.
        """
        ref_date = date.fromisoformat(reference_date)
        result, llm_succeeded = extract_goal_from_message(
            user_message, today=ref_date, context=extractor_context,
        )
        tool_ctx["extraction"] = result
        tool_ctx["extraction_llm_ok"] = llm_succeeded

        return json.dumps({
            "is_goal_intent": result.is_goal_intent,
            "name": result.name,
            "target_amount": result.target_amount,
            "target_date": (
                result.target_date.isoformat()
                if result.target_date else None
            ),
            "current_amount": result.current_amount,
            "missing_fields": result.missing_fields,
        }, default=str)

    @tool
    def create_goal_tool(
        name: str,
        target_amount: float,
        target_date: str,
        current_amount: float = 0.0,
    ) -> str:
        """
        Create a new FinancialGoal with the given parameters.

        The goal's required monthly saving and on-track status will be
        calculated automatically.  This tool validates that the target
        amount is positive and the target date is in the future.

        Returns the created goal as JSON.
        """
        parsed_date = date.fromisoformat(target_date)

        if target_amount <= 0:
            return "Error: target_amount must be positive."
        if parsed_date <= today:
            return (
                f"Error: target_date {target_date} must be in the future. "
                f"Today is {today.isoformat()}."
            )

        goal = FinancialGoal(
            id=f"goal-{uuid4().hex[:8]}",
            name=name or "Financial Goal",
            target_amount=float(target_amount),
            current_amount=float(current_amount),
            target_date=parsed_date,
        )
        tool_ctx["new_goal"] = goal
        tool_ctx["new_goal_added"] = True

        return json.dumps(goal.model_dump(mode="json"), default=str)

    @tool
    def calculate_required_saving_tool(
        target_amount: float,
        current_amount: float,
        target_date_str: str,
    ) -> str:
        """
        Calculate how much needs to be saved each month to reach a goal.

        Provide target_amount, current_amount (0 if none saved yet), and
        target_date (YYYY-MM-DD).  Returns the required monthly saving
        amount.  Useful for checking if a goal is on track.
        """
        parsed_date = date.fromisoformat(target_date_str)
        dummy_goal = FinancialGoal(
            id="calc",
            name="calc",
            target_amount=float(target_amount),
            current_amount=float(current_amount),
            target_date=parsed_date,
        )
        required = _calc_saving(dummy_goal)
        return f"Required monthly saving: {required:.2f}"

    # --- Build LLM with tools ---
    model_name = resolve_model_name()
    llm = ChatAnthropic(
        model=model_name, timeout=30
    ).bind_tools([
        extract_goal_tool,
        create_goal_tool,
        calculate_required_saving_tool,
        shared_final_answer,
    ])

    tools_map: dict[str, Any] = {
        "extract_goal_tool": extract_goal_tool,
        "create_goal_tool": create_goal_tool,
        "calculate_required_saving_tool": calculate_required_saving_tool,
    }

    existing_goals_summary = "\n".join(
        f"  {g.name}: target={g.target_amount}, "
        f"saved={g.current_amount}, "
        f"by={g.target_date.isoformat()}"
        for g in goals
    ) or "  (no existing goals)"

    system_text = get_react_prompt(
        "react_goal_planning",
        today=today.isoformat(),
        monthly_surplus=f"{monthly_surplus:.2f}",
        existing_goals=existing_goals_summary,
    )

    # Append relevant user history from long-term memory, if available.
    memory_ctx = view.get("memory_context") or ""
    if memory_ctx:
        system_text += f"\n\n## User History\n{memory_ctx}"

    react_msg = latest_message
    if extractor_context:
        react_msg = (
            f"[Continuing goal conversation. Context: {extractor_context}]\n\n"
            f"{react_msg}"
        )

    run_react_loop(
        llm=llm,
        tools=tools_map,
        system_prompt=system_text,
        user_message=(
            f"User message: {react_msg}\n\n"
            f"Financial context:\n"
            f"- Today: {today.isoformat()}\n"
            f"- Monthly income: {monthly_income:.2f}\n"
            f"- Monthly surplus: {monthly_surplus:.2f}\n"
            f"- Existing goals:\n{existing_goals_summary}"
        ),
        tool_ctx=tool_ctx,
    )

    # ------------------------------------------------------------------
    # Post-processing: build state update from tool_ctx
    # ------------------------------------------------------------------
    extraction: GoalExtractionResult | None = tool_ctx.get("extraction")
    new_goal_added = tool_ctx.get("new_goal_added", False)

    # Post-processing fallback: if ReAct loop didn't extract, try directly
    if extraction is None and latest_message.strip():
        result, ok = extract_goal_from_message(
            latest_message, today=today, context=extractor_context,
        )
        tool_ctx["extraction"] = result
        tool_ctx["extraction_llm_ok"] = ok
        extraction = result

    # ------------------------------------------------------------------
    # Update intent — modify an existing goal instead of creating new
    # ------------------------------------------------------------------
    if (
        extraction is not None
        and extraction.is_update_intent
        and goals
        and not new_goal_added
    ):
        # Try to match extraction name against existing goal names
        target_name = (extraction.name or "").lower()
        matched_goal = None
        for g in goals:
            g_name = g.name.lower()
            if target_name and (target_name in g_name or g_name in target_name):
                matched_goal = g
                break

        # If no name match, try fuzzy-match against the user message
        if matched_goal is None:
            msg_lower = latest_message.lower()
            for g in sorted(goals, key=lambda x: len(x.name), reverse=True):
                g_name = g.name.lower()
                # Check if any word from the goal name appears in the message
                name_words = [w for w in g_name.split() if len(w) > 2]
                if any(w in msg_lower for w in name_words):
                    matched_goal = g
                    break

        if matched_goal is not None:
            logger.debug(
                "[goal_planning] Update intent matched goal '%s'", matched_goal.name
            )
            updated = matched_goal.model_copy()
            if extraction.target_amount is not None:
                updated.target_amount = extraction.target_amount
            if extraction.target_date is not None:
                updated.target_date = extraction.target_date
            if extraction.current_amount is not None:
                updated.current_amount = extraction.current_amount
            if extraction.name:
                updated.name = extraction.name

            tool_ctx["new_goal"] = updated
            tool_ctx["new_goal_added"] = True
            tool_ctx["updated_goal_id"] = matched_goal.id
            new_goal_added = True
            # Replace the old goal with the updated one in the goals list
            goals = [updated if g.id == matched_goal.id else g for g in goals]

    # ReAct fallback: extraction succeeded but LLM skipped create_goal_tool
    # (e.g. LLM went straight to final_answer after extracting).
    # Create the goal directly from extraction results.
    if (
        extraction is not None
        and extraction.is_goal_intent
        and not extraction.is_update_intent
        and not extraction.missing_fields
        and not new_goal_added
        and extraction.target_amount is not None
        and extraction.target_date is not None
    ):
        logger.warning(
            "[goal_planning] ReAct loop skipped create_goal_tool "
            "- creating goal from extraction directly"
        )
        goal = FinancialGoal(
            id=f"goal-{uuid4().hex[:8]}",
            name=extraction.name or "Financial Goal",
            target_amount=extraction.target_amount,
            current_amount=extraction.current_amount or 0.0,
            target_date=extraction.target_date,
        )
        tool_ctx["new_goal"] = goal
        tool_ctx["new_goal_added"] = True
        new_goal_added = True

    # Handle clarification path (missing goal fields)
    if extraction and extraction.is_goal_intent and extraction.missing_fields:
        pending_confirmation = {
            "action": "clarify_goal_planning",
            "agent": "goal_planning",
            "summary": (
                "I detected a financial goal request, but some required "
                "information is missing."
            ),
            "details": [
                f"Detected goal name: {extraction.name or 'N/A'}",
                f"Missing fields: {', '.join(extraction.missing_fields)}",
                "Please confirm or provide the missing details.",
            ],
            "goal_extraction_confidence": (
                "llm" if tool_ctx.get("extraction_llm_ok") else "fallback"
            ),
        }
        # Don't add the goal since fields are missing
        clarification_msg = (
            f"I detected a financial goal request, but some information is missing. "
            f"Goal name: {extraction.name or 'N/A'}. "
            f"Missing: {', '.join(extraction.missing_fields)}."
        )
        if extraction.target_amount is not None:
            clarification_msg += f" Target amount: {extraction.target_amount}."
        return {
            "goals": goals,
            "pending_confirmation": pending_confirmation,
            "messages": [AIMessage(content=clarification_msg)],
        }

    # If a new goal was created, add it to the list.
    # Skip append for updates — the goal was already replaced in-place.
    updated_goals = list(goals)
    new_goal = tool_ctx.get("new_goal")
    if new_goal and new_goal_added and not tool_ctx.get("updated_goal_id"):
        updated_goals.append(new_goal)

    # Evaluate all goals
    final_goals: list[FinancialGoal] = []
    detail_lines: list[str] = []
    creation_lines: list[str] = []

    for goal in updated_goals:
        required = _calc_saving(goal)
        on_track = required <= monthly_surplus
        final_goal = goal.model_copy(
            update={
                "required_monthly_saving": required,
                "on_track": on_track,
            }
        )
        final_goals.append(final_goal)

        status = "on track" if on_track else "behind schedule"
        line = (
            f"{goal.name}: need to save {required:.2f}/month, "
            f"status = {status}"
        )
        detail_lines.append(line)
        if new_goal_added and goal.id == (new_goal.id if new_goal else None):
            creation_lines.append(line)

    # Build HITL confirmation
    summary_text = tool_ctx.get(
        "summary",
        (
            f"{'Created and e' if new_goal_added else 'E'}valuated "
            f"{len(final_goals)} financial goal(s). "
            f"Monthly surplus = {monthly_surplus:.2f}."
        ),
    )

    pending_confirmation = {}
    if new_goal_added:
        pending_confirmation = {
            "action": "approve_goal_planning",
            "agent": "goal_planning",
            "summary": summary_text,
            "details": creation_lines + detail_lines,
            "goal_extraction_confidence": (
                "llm"
                if tool_ctx.get("extraction_llm_ok")
                else "fallback"
            ),
        }
    else:
        # No new goal, just evaluation — no HITL needed
        pending_confirmation = {
            "action": "summarise_goals",
            "agent": "goal_planning",
            "summary": summary_text,
            "details": detail_lines,
        }

    return {
        "goals": final_goals,
        "pending_confirmation": (
            pending_confirmation
            if pending_confirmation.get("action") == "approve_goal_planning"
            else None
        ),
        "hitl_rollback": (
            {"original_goal_ids": original_goal_ids}
            if new_goal_added
            else None
        ),
        "messages": [AIMessage(content=summary_text)],
    }
