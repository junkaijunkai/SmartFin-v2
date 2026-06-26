"""
LangGraph Supervisor graph — wires all nodes together and compiles the graph.

Responsibilities:
  - Define each node (supervisor + 5 specialist agents + HITL confirm node).
  - Add edges and conditional edges using the router functions.
  - Compile with a checkpointer so state is persisted after every step,
    enabling HITL interrupts and session resumption.

Each specialist agent node is a STUB here (returns state unchanged).
Replace the stub functions with real imports as each agent is implemented.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from app.state import AppState
from app.orchestrator.checkpoints import memory_checkpointer
from app.orchestrator.intent_classifier import classify_intent
from app.orchestrator.state_view import AgentStateView, AGENT_SCOPES
from app.orchestrator.context import compress_messages
from app.orchestrator.router import (
    NODE_ANOMALY_DETECTION,
    NODE_BUDGET_PLANNING,
    NODE_CONFIRM,
    NODE_EXPENSE_ANALYSIS,
    NODE_GOAL_PLANNING,
    NODE_HEALTH_ASSESSMENT,
    NODE_MEMORY_LOADER,
    NODE_MEMORY_SAVER,
    NODE_SUPERVISOR,
    route_after_agent,
    route_to_agent,
)
from app.tools.memory_store import MemoryStore
from app.tools.transaction_store import save_analysis, save_user_analysis
from app.observability import enter_span, log_trace_event, set_thread_id
from app.observability.events import (SUPERVISOR_DECISION, TRACE_STEP, STATE_SNAPSHOT,
                                      ERROR_CATEGORISED, VALIDATION_ERROR)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node implementations (stubs — replace with real agent imports)
# ---------------------------------------------------------------------------


def memory_loader_node(state: AppState, config: RunnableConfig) -> dict:
    """
    Minimal loader — sets trace context and retrieves relevant long-term memory.

    Does NOT strip messages, load working memory, or conditionally load
    from filesystem.  Checkpoint is the source of truth for all state fields.
    """
    with enter_span("memory_loader"):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if not thread_id:
            return {}

        set_thread_id(thread_id)

        result: dict = {}

        # Retrieve relevant long-term memory based on the latest user message.
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            user_text = getattr(last_msg, "content", "") or ""
            if user_text:
                from app.memory.retriever import retrieve_memory
                try:
                    memories = retrieve_memory(user_text)
                    if memories:
                        result["memory_context"] = "\n\n".join(
                            m["content"] for m in memories
                        )
                except Exception as exc:
                    logger.debug("[memory] Retrieval failed: %s", exc)

        # Restore categorised transactions from user-level cache when this is a
        # fresh thread (checkpoint has no categorised data yet).
        if not state.get("categorised_transactions"):
            from app.tools.transaction_store import load_user_analysis
            cached = load_user_analysis()
            if cached:
                cats, trends = cached
                result["categorised_transactions"] = cats
                result["spending_trends"] = trends
                logger.debug(
                    "[memory_loader] restored %d categorised transactions from user cache",
                    len(cats),
                )

        log_trace_event(
            STATE_SNAPSHOT,
            active_agent=state.get("active_agent"),
            pending_intent=state.get("pending_intent"),
            categorised_count=len(state.get("categorised_transactions") or []),
            transactions_count=len(state.get("transactions") or []),
            has_memory_context="memory_context" in result,
        )
        return result


def memory_saver_node(state: AppState, config: RunnableConfig) -> dict:
    """
    Persist structured state back to the filesystem after an agent finishes,
    and update the conversation summary in working memory.

    When ``pending_confirmation.action`` starts with ``"approve_"``, the
    agent's writes are **tentative** — they should NOT be persisted until
    the user approves them via HITL.  In that case we skip ``save_state()``
    but still save working memory (conversation summaries are accumulative).
    """
    with enter_span("memory_saver"):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if not thread_id:
            return {}

        store = MemoryStore(thread_id)

        # Skip persisting agent writes that are waiting for HITL approval.
        pending = state.get("pending_confirmation") or {}
        is_tentative = (
            isinstance(pending, dict)
            and isinstance(pending.get("action"), str)
            and pending["action"].startswith("approve_")
        )

        if not is_tentative:
            store.save_state(dict(state))

        # Write confirmed agent outputs to long-term memory.
        if not is_tentative:
            _write_long_term_memory(thread_id, state)

        log_trace_event(
            STATE_SNAPSHOT,
            is_tentative=is_tentative,
            active_agent=state.get("active_agent"),
            pending_intent=state.get("pending_intent"),
        )
        return {}  # no state update — all effects are filesystem writes


def _write_long_term_memory(thread_id: str, state: AppState) -> None:
    """Persist confirmed agent outputs to the vector memory store."""
    from app.memory.writer import write_task_memory
    agent_name = state.get("active_agent")
    if not agent_name:
        return
    write_task_memory(agent_name, dict(state))


def supervisor_node(state: AppState, config: RunnableConfig | None = None) -> dict:
    """
    Routes user requests to appropriate worker agents.

    Loop continuation pattern — decides ONE agent at a time.
    When expense_analysis must run first, stores the original intent
    in ``pending_intent`` and re-routes after prep completes.

    Branch A — loop continuation (an agent just finished):
      - pending_intent is set and ≠ active_agent → prep scenario
        - categorised data exists → route to pending_intent
        - no categorised data → end (prep failed)
      - else → target reached, end.

    Branch B — fresh user message:
      - classify intent → check data → route to ONE agent.
    """
    with enter_span("supervisor"):
        active = state.get("active_agent")
        pending = state.get("pending_intent")

        # =================================================================
        # Branch A: Agent just completed — decide next step
        # =================================================================
        if active not in (None, "end"):
            log_trace_event(
                TRACE_STEP,
                action="branch_a",
                active_agent=active,
                pending_intent=pending,
                categorised_count=len(state.get("categorised_transactions") or []),
            )

            if pending and pending != active:
                # Prep scenario: expense_analysis ran for a pending target
                cats = state.get("categorised_transactions")

                if active == "expense_analysis" and cats:
                    with enter_span("route_to_pending"):
                        log_trace_event(
                            SUPERVISOR_DECISION,
                            action="route_to_pending",
                            from_agent=active,
                            to_agent=pending,
                            categorised_count=len(state.get("categorised_transactions") or []),
                        )
                    return {"active_agent": pending, "pending_intent": None}
                # Prep failed or unexpected — end safely
                log_trace_event(
                    SUPERVISOR_DECISION,
                    action="end_prep_failed",
                    active_agent=active,
                    pending_intent=pending,
                    has_categorised=bool(state.get("categorised_transactions")),
                )
                return {"active_agent": "end", "pending_intent": None}
            # Target reached or direct expense_analysis request
            log_trace_event(
                SUPERVISOR_DECISION,
                action="end_target_reached",
                active_agent=active,
                pending_intent=pending,
            )
            return {"active_agent": "end", "pending_intent": None}

    # =================================================================
    # Branch B: Fresh turn — classify and route to one agent
    # =================================================================
    with enter_span("supervisor"):
        messages = state.get("messages", [])

        last_message = messages[-1].content if messages else ""

        # Gateway guardrail handles injection detection before the LLM call.
        # classify_intent() returns "blocked" if the gateway rejects the request.
        agent_name = classify_intent(last_message)

        if agent_name == "blocked":
            from langchain_core.messages import AIMessage
            return {
                "active_agent": "end",
                "pending_intent": None,
                "messages": [AIMessage(content=(
                    "Your request was blocked because it appears to contain unsafe or "
                    "irrelevant instructions. Please submit a normal personal finance query."
                ))],
            }

        # When intent is unknown but the conversation is a continuation of a
        # previous turn (e.g. the user is clarifying a goal or budget), fall
        # back to the prior agent instead of ending the conversation.
        if agent_name == "unknown":
            prev = state.get("last_intent")
            if prev and prev not in ("unknown", "end"):
                agent_name = prev

        if agent_name == "unknown":
            from langchain_core.messages import AIMessage
            return {
                "active_agent": "end",
                "pending_intent": None,
                "input_filter_result": filter_result,
                "messages": [AIMessage(content=(
                    '''
                    I'm SmartFin, your personal finance AI assistant.
                    I can only help you with:

                      - **Expense Analysis** — break down your spending by category and spot trends
                      - **Budget Planning** — set and review monthly spending limits
                      - **Goal Planning** — create and track savings goals (e.g. emergency fund, holiday)
                      - **Anomaly Detection** — flag suspicious or unusual transactions
                      - **Financial Health Assessment** — get an overall picture of your financial health

                    Please rephrase your request to fit one of these categories, and I'll do my best to assist you!
                    '''
                ))],
            }

        # --- Data availability check ---
        # memory_loader has already loaded any persisted data from MemoryStore
        has_categorised = bool(state.get("categorised_transactions"))
        has_new_txns = bool(state.get("transactions"))

        # Log the full routing decision
        log_trace_event(
            SUPERVISOR_DECISION,
            action="branch_b",
            intent=agent_name,
            has_categorised=has_categorised,
            has_new_txns=has_new_txns,
        )

        # Record classified intent for next turn (replaces working_memory.conversation_summary)
        last_intent = agent_name

        if has_categorised:
            # Data already available → route directly to target agent
            return {
                "active_agent": agent_name,
                "pending_intent": None,
                "input_filter_result": filter_result,
                "last_intent": last_intent,
            }

        if has_new_txns:
            # Need expense_analysis first
            if agent_name == "expense_analysis":
                # User directly asked for expense_analysis — route directly
                return {
                    "active_agent": agent_name,
                    "pending_intent": agent_name,
                    "input_filter_result": filter_result,
                    "last_intent": last_intent,
                }
            # Prepend expense_analysis, store the real target as pending
            return {
                "active_agent": "expense_analysis",
                "pending_intent": agent_name,
                "input_filter_result": filter_result,
                "last_intent": last_intent,
            }

        # No transaction data available at all
        if agent_name == "expense_analysis":
            # Let the agent try to extract data from the user's message
            return {
                "active_agent": agent_name,
                "pending_intent": None,
                "input_filter_result": filter_result,
                "last_intent": last_intent,
            }

        from langchain_core.messages import AIMessage
        return {
            "active_agent": "end",
            "pending_intent": None,
            "messages": [AIMessage(content=(
                "I need transaction data to help you. "
                "Please provide your recent transactions so I can get started."
            ))],
        }


def expense_analysis_node(state: AppState) -> dict:
    from app.agents.expense_analysis.agent import run
    return run(state) 


def budget_planning_node(state: AppState) -> dict:
    from app.agents.budget_planning.agent import budget_planning_node as run_budget_planning
    return run_budget_planning(state)


# def goal_planning_node(state: AppState) -> dict:
#     """Stub — to be replaced by smartfin.agents.goal_planning.agent"""
#     print("[stub] goal_planning_node called")
#     return {}
def goal_planning_node(state: AppState) -> dict:
    """
    Goal Planning node.

    Calls the real Goal Planning agent, which:
    - extracts goal information from the user's message
    - creates/evaluates goals
    - returns pending_confirmation for HITL
    """
    from app.agents.goal_planning.agent import run
    return run(state)


def anomaly_detection_node(state: AppState) -> dict:
    from app.agents.anomaly_detection.agent import run
    return run(state) # anomaly_detection can be called standalone or after expense_analysis


def health_assessment_node(state: AppState) -> dict:
    from app.agents.health_assessment.agent import run
    return run(state)


def confirm_node(state: AppState, config: RunnableConfig | None = None) -> dict:
    """
    HITL confirmation node.

    The graph is compiled with interrupt_before=[NODE_CONFIRM], so execution
    pauses BEFORE this node runs.  The UI reads state["pending_confirmation"],
    presents it to the user, then resumes with {"pending_confirmation": {"confirmed": bool}}.

    Three scenarios:
      1. Approved: for expense_analysis, read categorised data directly from state
         (already written by the agent) and persist cache. For other agents, commit
         hitl_rollback as before.
      2. Rejected: for expense_analysis, actively zero out the state fields written
         by the agent. For other agents, discard hitl_rollback. Clear pending_intent.
      3. Clarification: user typed free text — re-route to supervisor without committing.
    """
    with enter_span("confirm"):
        confirmation = state.get("pending_confirmation", {})
        action = confirmation.get("action")
        hitl_decision = state.get("hitl_decision") or {}
        confirmed = hitl_decision.get("confirmed")

        if confirmed:
            # Clarification path — active_agent is None when the UI sends a free-text clarification.
            if state.get("active_agent") is None and len(state.get("messages", [])) > 0:
                log_trace_event(TRACE_STEP, action="clarification", confirmed=True)
                return {"pending_confirmation": None, "hitl_decision": None}

            log_trace_event(TRACE_STEP, action=action, confirmed=True,
                            pending_intent=state.get("pending_intent"))

            if action == "approve_expense_analysis":
                thread_id = (config or {}).get("configurable", {}).get("thread_id")
                if thread_id:
                    cats = state.get("categorised_transactions") or []
                    trends = state.get("spending_trends") or []
                    if cats:
                        save_analysis(thread_id, cats, trends)
                        save_user_analysis(cats, trends)

                    # Persist approved data so memory_loader can restore it.
                    store = MemoryStore(thread_id)
                    store.save_state(dict(state))
                    _write_long_term_memory(thread_id, state)
                return {"pending_confirmation": None, "hitl_rollback": None, "hitl_decision": None}

            if action == "approve_goal_planning":
                thread_id = (config or {}).get("configurable", {}).get("thread_id")
                if thread_id:
                    # memory_saver skipped save_state() (tentative writes),
                    # so we persist the approved goal here.
                    store = MemoryStore(thread_id)
                    store.save_state(dict(state))
                    _write_long_term_memory(thread_id, state)
                return {"pending_confirmation": None, "hitl_rollback": None, "hitl_decision": None}

            # Other agents still use hitl_rollback staging.
            staged = state.get("hitl_rollback") or {}
            return {**staged, "pending_confirmation": None, "hitl_rollback": None, "hitl_decision": None}

        else:
            log_trace_event(TRACE_STEP, action=action, confirmed=False,
                            pending_intent=state.get("pending_intent"))
            if action == "approve_expense_analysis":
                # Agent already wrote to typed state fields — actively clear them on reject.
                return {
                    "categorised_transactions": [],
                    "spending_trends": [],
                    "expense_analysis": {},
                    "pending_confirmation": None,
                    "hitl_rollback": None,
                    "hitl_decision": None,
                    "pending_intent": None,
                }
            if action == "approve_goal_planning":
                # Restore goals to pre-agent snapshot by filtering out any newly added goal.
                original_ids = set(
                    (state.get("hitl_rollback") or {}).get("original_goal_ids") or []
                )
                restored = [g for g in (state.get("goals") or []) if g.id in original_ids]
                return {
                    "goals": restored,
                    "pending_confirmation": None,
                    "hitl_rollback": None,
                    "hitl_decision": None,
                    "pending_intent": None,
                }
            return {"pending_confirmation": None, "hitl_rollback": None, "hitl_decision": None, "pending_intent": None}


# ---------------------------------------------------------------------------
# Agent wrapper — scoped state view + write validation + message compression
# ---------------------------------------------------------------------------


def _wrap_agent(agent_name: str, agent_fn):
    """
    Wrap an agent node with:
      1. ``AgentStateView`` — only exposes fields the agent is allowed to read.
      2. Write validation — raising ``ValueError`` if the agent tries to write
         a field outside its declared scope.
      3. Message compression — runs ``compress_messages`` after the agent
         finishes, so long-running conversations stay within budget.
    """

    def wrapped(state: AppState, config=None) -> dict:
        with enter_span(agent_name):
            view = AgentStateView(state, agent_name)

            # Run the agent (receives view, returns state updates)
            result = agent_fn(view)

            # -- Validate writes --
            allowed_writes = AGENT_SCOPES[agent_name]["writes"]
            for key in result:
                if key not in allowed_writes:
                    log_trace_event(ERROR_CATEGORISED, error_category=VALIDATION_ERROR,
                                    agent=agent_name, field=key,
                                    allowed=list(allowed_writes))
                    raise ValueError(
                        f"Agent '{agent_name}' wrote to disallowed field "
                        f"'{key}'. Allowed writes: {sorted(allowed_writes)}"
                    )

            # -- Compress messages if they grew --
            if "messages" in result:
                existing = list(state.get("messages", []) or [])
                new_msgs = list(result["messages"] or [])
                combined = existing + new_msgs
                result["messages"] = compress_messages(
                    combined, agent_name=agent_name
                )

            log_trace_event(
                TRACE_STEP,
                writes=list(result.keys()),
                categorised_count=len(result.get("categorised_transactions") or []),
                has_pending=bool(result.get("pending_confirmation")),
            )
            return result

    return wrapped


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Construct and compile the SmartFin StateGraph.

    Returns a compiled graph ready to invoke:
        app = build_graph()
        config = {"configurable": {"thread_id": "user-123"}}
        result = app.invoke({"messages": [HumanMessage(content="...")]}, config)
    """
    builder = StateGraph(AppState)

    # --- Register nodes ---
    builder.add_node(NODE_MEMORY_LOADER, memory_loader_node)
    builder.add_node(NODE_SUPERVISOR, supervisor_node)
    builder.add_node(
        NODE_EXPENSE_ANALYSIS,
        _wrap_agent(NODE_EXPENSE_ANALYSIS, expense_analysis_node),
    )
    builder.add_node(
        NODE_BUDGET_PLANNING,
        _wrap_agent(NODE_BUDGET_PLANNING, budget_planning_node),
    )
    builder.add_node(
        NODE_GOAL_PLANNING,
        _wrap_agent(NODE_GOAL_PLANNING, goal_planning_node),
    )
    builder.add_node(
        NODE_ANOMALY_DETECTION,
        _wrap_agent(NODE_ANOMALY_DETECTION, anomaly_detection_node),
    )
    builder.add_node(
        NODE_HEALTH_ASSESSMENT,
        _wrap_agent(NODE_HEALTH_ASSESSMENT, health_assessment_node),
    )
    builder.add_node(NODE_MEMORY_SAVER, memory_saver_node)
    builder.add_node(NODE_CONFIRM, confirm_node)

    # --- Entry: every turn starts by loading state from filesystem ---
    builder.set_entry_point(NODE_MEMORY_LOADER)

    # --- Memory loader → supervisor ---
    builder.add_edge(NODE_MEMORY_LOADER, NODE_SUPERVISOR)

    # --- Supervisor routes conditionally to one of the specialist agents ---
    builder.add_conditional_edges(
        NODE_SUPERVISOR,
        route_to_agent,
        {
            NODE_EXPENSE_ANALYSIS:  NODE_EXPENSE_ANALYSIS,
            NODE_BUDGET_PLANNING:   NODE_BUDGET_PLANNING,
            NODE_GOAL_PLANNING:     NODE_GOAL_PLANNING,
            NODE_ANOMALY_DETECTION: NODE_ANOMALY_DETECTION,
            NODE_HEALTH_ASSESSMENT: NODE_HEALTH_ASSESSMENT,
            NODE_CONFIRM:           NODE_CONFIRM,
            END:                    END,
        },
    )

    # --- After each agent, persist state to filesystem ---
    for agent_node in [
        NODE_EXPENSE_ANALYSIS,
        NODE_BUDGET_PLANNING,
        NODE_GOAL_PLANNING,
        NODE_ANOMALY_DETECTION,
        NODE_HEALTH_ASSESSMENT,
    ]:
        builder.add_edge(agent_node, NODE_MEMORY_SAVER)

    # --- Memory saver routes: confirm (if HITL) or memory_loader (next turn) ---
    builder.add_conditional_edges(
        NODE_MEMORY_SAVER,
        route_after_agent,
        {
            NODE_CONFIRM:       NODE_CONFIRM,
            NODE_MEMORY_LOADER: NODE_MEMORY_LOADER,
        },
    )

    # --- After confirm node, load state again for the next turn ---
    builder.add_edge(NODE_CONFIRM, NODE_MEMORY_LOADER)

    # --- Compile with checkpointer for state persistence and HITL support ---
    graph = builder.compile(
        checkpointer=memory_checkpointer,
        interrupt_before=[NODE_CONFIRM],  # pause before confirm node for HITL
    )

    return graph
