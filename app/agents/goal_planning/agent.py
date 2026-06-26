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
from typing import Annotated, Any, Optional
from pydantic import Field
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.orchestrator.state_view import AgentStateView
from app.agents.react_utils import run_react_loop
from app.agents.goal_planning.tracker import calculate_required_monthly_saving as _calc_saving
from app.agents.goal_planning.extractor import extract_goal_from_message, GoalExtractionResult
from app.config import get_llm, get_react_prompt
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
    def create_goal_tool(
        goal_name: Annotated[
            Optional[str],
            Field(description=(
                "Goal name inferred from the item mentioned. "
                "Convert item references to fund names: 'a Mercedes' → 'Mercedes Fund', "
                "'an iPhone' → 'iPhone Fund', 'a house deposit' → 'House Deposit Fund'. "
                "Set to null only if no item or purpose is mentioned."
            )),
        ],
        target_amount: Annotated[
            Optional[float],
            Field(description=(
                "Target savings amount as a positive number. "
                "Set to null only if the user has not mentioned an amount."
            )),
        ],
        target_date_iso: Annotated[
            Optional[str],
            Field(description=(
                "Target date as YYYY-MM-DD. Resolve date expressions in your reasoning "
                "before calling: 'by the end of 2026' → '2026-12-31', "
                "'by June' → last day of June, "
                "'in 3 months' → last day of the month 3 months from today, "
                "'next year' → December 31 of next year. "
                "Set to null only if no date is mentioned."
            )),
        ],
        current_amount: Annotated[
            float,
            Field(description="Amount already saved toward this goal. Use 0.0 if not mentioned."),
        ] = 0.0,
        is_update: Annotated[
            bool,
            Field(description=(
                "Set to true when the user wants to modify an existing goal "
                "(e.g. 'update my Mercedes goal to $60000'). "
                "goal_name will be matched against existing goals by name."
            )),
        ] = False,
    ) -> str:
        """
        Validate goal parameters and create or update a FinancialGoal in a single step.

        Call this once you have extracted all goal parameters from the user's message
        in your own reasoning. Resolve all date expressions to YYYY-MM-DD before calling.
        Do not call this tool more than once per user turn.

        Behaviour:
        - If target_amount or target_date_iso is null (and is_update is false): returns
          status 'missing_fields'. Do NOT call final_answer yet — tell the user which
          fields are missing and ask for them.
        - If all required fields are present and is_update is false: creates a new
          FinancialGoal and returns status 'created'.
        - If is_update is true: finds the matching existing goal by name and applies
          provided updates, returning status 'updated'. If no match is found, returns
          status 'error' with available goal names.

        Returns JSON with 'status' ('created', 'updated', 'missing_fields', or 'error').

        Do NOT call this tool for query intents (e.g. "what is my goal",
        "show me my goals", "how are my savings going"). For queries, use the
        existing goals listed in the system prompt context and produce an
        end_turn response directly.
        """
        missing: list[str] = []
        if not is_update:
            if target_amount is None:
                missing.append("target_amount")

        parsed_date: Optional[date] = None
        if target_date_iso is not None:
            try:
                parsed_date = date.fromisoformat(target_date_iso)
            except ValueError:
                if not is_update:
                    missing.append("target_date")
        elif not is_update:
            missing.append("target_date")

        # Always store extraction so post-processing / clarification path can read it
        extraction_result = GoalExtractionResult(
            is_goal_intent=True,
            name=goal_name,
            target_amount=target_amount,
            target_date=parsed_date,
            current_amount=current_amount or 0.0,
            missing_fields=missing,
            is_update_intent=is_update,
        )
        tool_ctx["extraction"] = extraction_result
        tool_ctx["extraction_llm_ok"] = True

        if missing:
            return json.dumps({
                "status": "missing_fields",
                "missing_fields": missing,
                "extracted_so_far": {
                    "goal_name": goal_name,
                    "target_amount": target_amount,
                    "target_date_iso": target_date_iso,
                },
            })

        if is_update:
            # Find matching goal by name (exact → substring → fuzzy word match)
            target_name = (goal_name or "").lower()
            matched: Optional[FinancialGoal] = None
            for g in goals:
                g_lower = g.name.lower()
                if target_name and (target_name in g_lower or g_lower in target_name):
                    matched = g
                    break
            if matched is None:
                msg_lower = latest_message.lower()
                for g in sorted(goals, key=lambda x: len(x.name), reverse=True):
                    if any(w in msg_lower for w in g.name.lower().split() if len(w) > 2):
                        matched = g
                        break

            if matched is None:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"No existing goal found matching '{goal_name}'. "
                        f"Available goals: {[g.name for g in goals]}"
                    ),
                })

            updated = matched.model_copy()
            if target_amount is not None:
                updated.target_amount = target_amount
            if parsed_date is not None:
                updated.target_date = parsed_date
            if current_amount:
                updated.current_amount = current_amount
            if goal_name:
                updated.name = goal_name

            # In-place update so post-processing sees the modified list
            for i, g in enumerate(goals):
                if g.id == matched.id:
                    goals[i] = updated
                    break

            tool_ctx["new_goal"] = updated
            tool_ctx["new_goal_added"] = True
            tool_ctx["updated_goal_id"] = matched.id

            return json.dumps({
                "status": "updated",
                "goal": updated.model_dump(mode="json"),
            }, default=str)

        # Create new goal
        if target_amount <= 0:
            return json.dumps({"status": "error", "message": "target_amount must be positive."})
        if parsed_date <= today:
            return json.dumps({
                "status": "error",
                "message": (
                    f"target_date {target_date_iso} must be in the future "
                    f"(today is {today.isoformat()})."
                ),
            })

        goal = FinancialGoal(
            id=f"goal-{uuid4().hex[:8]}",
            name=goal_name or "Financial Goal",
            target_amount=float(target_amount),
            current_amount=float(current_amount),
            target_date=parsed_date,
        )
        tool_ctx["new_goal"] = goal
        tool_ctx["new_goal_added"] = True

        return json.dumps({
            "status": "created",
            "goal": goal.model_dump(mode="json"),
        }, default=str)

    @tool
    def calculate_required_saving_tool(
        target_amount: Annotated[float, Field(description="The total savings target amount.")],
        current_amount: Annotated[float, Field(description="Amount already saved (use 0.0 if none).")],
        target_date_str: Annotated[str, Field(description="Target date as YYYY-MM-DD.")],
    ) -> str:
        """
        Calculate the monthly saving required to reach a financial goal on schedule.

        Call this after create_goal_tool returns status 'created' or 'updated'.
        Do not call this if create_goal_tool returned 'missing_fields' or 'error'.
        Use the goal parameters (amount, current savings, date) from the tool response.

        Returns the minimum monthly saving amount needed to meet the goal on time.
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
    llm = get_llm("planner", timeout=30).bind_tools([
        create_goal_tool,
        calculate_required_saving_tool,
    ])

    tools_map: dict[str, Any] = {
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

    # Build recent conversation history for multi-turn continuations
    # (e.g. user provides missing goal fields in a follow-up message)
    all_msgs = view.get("messages") or []
    history_msgs = [
        m for m in all_msgs
        if hasattr(m, 'type') and m.type in ('human', 'ai') and getattr(m, 'content', '')
    ][:-1]  # Exclude the latest message (already in user_message below)

    response, _ = run_react_loop(
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
        history=history_msgs[-6:] if history_msgs else None,
    )

    # ------------------------------------------------------------------
    # Post-processing: build state update from tool_ctx
    # ------------------------------------------------------------------
    extraction: GoalExtractionResult | None = tool_ctx.get("extraction")
    new_goal_added = tool_ctx.get("new_goal_added", False)

    # Fallback: create_goal_tool was never called (model non-compliance)
    if extraction is None and latest_message.strip():
        logger.warning("[goal_planning] create_goal_tool not called — running extraction fallback")
        result, ok = extract_goal_from_message(
            latest_message, today=today, context=extractor_context,
        )
        tool_ctx["extraction"] = result
        tool_ctx["extraction_llm_ok"] = ok
        extraction = result
        if (
            result.is_goal_intent
            and not result.is_update_intent
            and not result.missing_fields
            and result.target_amount is not None
            and result.target_date is not None
        ):
            goal = FinancialGoal(
                id=f"goal-{uuid4().hex[:8]}",
                name=result.name or "Financial Goal",
                target_amount=result.target_amount,
                current_amount=result.current_amount or 0.0,
                target_date=result.target_date,
            )
            tool_ctx["new_goal"] = goal
            tool_ctx["new_goal_added"] = True
            new_goal_added = True

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
    default_fallback = (
        f"{'Created and e' if new_goal_added else 'E'}valuated "
        f"{len(final_goals)} financial goal(s). "
        f"Monthly surplus = {monthly_surplus:.2f}."
    )
    raw_content = response.content
    if isinstance(raw_content, str):
        summary_text = raw_content.strip() or default_fallback
    elif isinstance(raw_content, list):
        # Claude sometimes returns content as a list of blocks; extract plain text.
        texts = [
            block.get("text", "") for block in raw_content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        summary_text = " ".join(texts).strip() or default_fallback
    else:
        summary_text = default_fallback

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
