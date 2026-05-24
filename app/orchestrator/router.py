"""
Supervisor routing logic.

The router is a plain function — NOT a class — that LangGraph calls via
add_conditional_edges() after the supervisor node runs.  It reads the
current AppState and returns the name of the next node to execute,
or END to terminate the graph.

Flow overview:
                          ┌─────────────────────────────────────┐
                          │           SUPERVISOR NODE            │
                          │  (decides intent from user message)  │
                          └────────────────┬────────────────────┘
                                           │ route_to_agent()
              ┌──────────────┬─────────────┼──────────────┬──────────────┐
              ▼              ▼             ▼              ▼              ▼
       expense_analysis  budget_planning  goal_planning  anomaly_detection  health_assessment
              │              │             │              │              │
              └──────────────┴─────────────┴──────────────┴──────────────┘
                                           │
                                    back to SUPERVISOR
                                    (or END if done)
"""

from langgraph.graph import END

from app.state import AppState

# Node name constants — single source of truth used by both router and graph.
NODE_SUPERVISOR = "supervisor"
NODE_EXPENSE_ANALYSIS = "expense_analysis"
NODE_BUDGET_PLANNING = "budget_planning"
NODE_GOAL_PLANNING = "goal_planning"
NODE_ANOMALY_DETECTION = "anomaly_detection"
NODE_HEALTH_ASSESSMENT = "health_assessment"
NODE_CONFIRM = "confirm"  # HITL confirmation node
NODE_MEMORY_LOADER = "memory_loader"
NODE_MEMORY_SAVER = "memory_saver"


def route_to_agent(state: AppState) -> str:
    """
    Called by add_conditional_edges after the supervisor node.

    Reads state["active_agent"] — a field the supervisor node writes to
    signal which specialist should run next — and maps it to a node name.

    Returns END when the supervisor signals that all required work is done.

    Note: the supervisor node is responsible for deciding WHICH agent is
    needed (e.g. by parsing the user's message with an LLM).  This function
    is purely a lookup; keep it free of LLM calls.
    """
    next_agent = state.get("active_agent")

    routing_map = {
        "expense_analysis":   NODE_EXPENSE_ANALYSIS,
        "budget_planning":    NODE_BUDGET_PLANNING,
        "goal_planning":      NODE_GOAL_PLANNING,
        "anomaly_detection":  NODE_ANOMALY_DETECTION,
        "health_assessment":  NODE_HEALTH_ASSESSMENT,
        "confirm":            NODE_CONFIRM,
        "end":                END,
        None:                 END,
    }

    return routing_map.get(next_agent, END)


def route_after_agent(state: AppState) -> str:
    """
    Called by add_conditional_edges after each specialist agent node.

    After an agent completes, we always return to the supervisor so it can
    decide whether another agent is needed or the conversation is done.
    In cases where the agent set pending_confirmation, we route to the
    HITL confirm node first.
    """
    if state.get("pending_confirmation") and not state["pending_confirmation"].get("confirmed"):
        return NODE_CONFIRM
    return NODE_MEMORY_LOADER
