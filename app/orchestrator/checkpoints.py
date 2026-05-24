"""
Checkpointer setup and Human-in-the-Loop (HITL) utilities.

Two distinct concepts live here:

1. Checkpointer  — LangGraph's built-in state persistence layer.
   After every node execution, LangGraph serialises the full AppState
   and stores it via the checkpointer.  This lets us:
     - Resume a paused graph (e.g. after HITL interrupt) without losing state.
     - Replay or inspect any past execution step.
     - Rewind to an earlier checkpoint and re-run (used by the UI's
       edit-and-resend feature).

   We persist checkpoints in a local SQLite file under .smartfin/
   so sessions survive server restarts.

2. HITL helpers — thin wrappers that the UI / CLI layer calls to
   resume a graph that has been paused by interrupt_before.
   The graph itself declares WHICH nodes trigger an interrupt
   (see graph.py); these helpers handle the resume flow.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpointer instance
# ---------------------------------------------------------------------------

# Register all Pydantic state models so LangGraph's JsonPlusSerializer doesn't
# emit a WARNING for each object when deserializing checkpointed state.
_STATE_MODULE = "app.state"
_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        (_STATE_MODULE, "Transaction"),
        (_STATE_MODULE, "BudgetAllocation"),
        (_STATE_MODULE, "FinancialGoal"),
        (_STATE_MODULE, "AnomalyFlag"),
        (_STATE_MODULE, "SpendingTrend"),
        (_STATE_MODULE, "HealthSummary"),
        (_STATE_MODULE, "Alert"),
        (_STATE_MODULE, "TransactionCategory"),
        (_STATE_MODULE, "AnomalyType"),
        (_STATE_MODULE, "HealthRating"),
        (_STATE_MODULE, "AlertSeverity"),
    ]
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / ".smartfin"

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://smartfin:smartfin@localhost:5432/smartfin",
)

def _init_checkpointer(max_attempts: int = 5, backoff: float = 2.0):
    for attempt in range(1, max_attempts + 1):
        try:
            conn = psycopg.connect(
                _DB_URL,
                autocommit=True,
                prepare_threshold=0,
                row_factory=dict_row,
            )
            checkpointer = PostgresSaver(conn, serde=_serde)
            checkpointer.setup()
            logger.info("Checkpointer initialised with PostgreSQL")
            return conn, checkpointer
        except Exception as exc:
            if attempt < max_attempts:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Failed to init PostgresSaver (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, max_attempts, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to init PostgresSaver after %d attempts: %s",
                    max_attempts, exc,
                )
                raise


_conn, memory_checkpointer = _init_checkpointer()


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------

def get_pending_interrupt(graph, config: dict) -> dict | None:
    """
    Return the current AppState if the graph is paused at an interrupt,
    or None if it has already finished.

    Usage:
        state = get_pending_interrupt(app_graph, config)
        if state:
            print("Waiting for user confirmation:", state["pending_confirmation"])
    """
    snapshot = graph.get_state(config)
    # snapshot.next is a tuple of node names that are about to execute.
    # If it's non-empty the graph is paused and waiting.
    if snapshot.next:
        return snapshot.values
    return None


def resume_with_confirmation(graph, config: dict, confirmed: bool, user_message: str | None = None) -> dict:
    """
    Resume a paused graph after the user has accepted, rejected, or provided
    additional information for the pending action.

    LangGraph resumes by calling graph.invoke(state_update, config).
    Passing None as the first argument means "use the existing checkpointed
    state"; we only need to patch the fields that changed.

    Args:
        graph:       The compiled StateGraph (returned by build_graph()).
        config:      The same thread config dict that was used to start the run.
                     Must contain {"configurable": {"thread_id": "..."}}.
        confirmed:   True = user approved the pending action,
                     False = user rejected/cancelled it.
        user_message: (Optional) When action is "clarify_*", user can provide
                     supplementary information here. The message is appended
                     to the messages list and the graph re-routes to the
                     original agent with new context.

    Returns:
        The final AppState dict after the graph resumes and finishes.

    Example usage for clarification:
        >>> # User rejected missing fields prompt and now provides clarification
        >>> resume_with_confirmation(
        ...     graph, config,
        ...     confirmed=True,  # confirmed=True indicates "user provided info"
        ...     user_message="I want to save $8000 by June 2027"
        ... )
    """
    
    from langchain_core.messages import HumanMessage

    # Preserve the original pending_confirmation fields (action, agent, summary,
    # details) so confirm_node can dispatch correctly. The confirmed flag is
    # written into a SEPARATE hitl_decision dict — NOT into pending_confirmation.
    #
    # Why: route_after_agent checks pending_confirmation.confirmed; if we put
    # confirmed=True inside pending_confirmation before graph.update_state, the
    # resulting fork causes route_after_agent to re-evaluate and skip straight
    # to supervisor_node, bypassing confirm_node entirely.  A separate field
    # keeps the routing intact.
    snapshot = graph.get_state(config)
    current_pc = {}
    if snapshot and snapshot.values:
        current_pc = dict(snapshot.values.get("pending_confirmation") or {})

    update = {
        "pending_confirmation": current_pc,
        "hitl_decision": {"confirmed": confirmed},
    }

    if user_message and confirmed:
        update["messages"] = [HumanMessage(content=user_message)]
        update["active_agent"] = None

    graph.update_state(config, update)
    return graph.invoke(None, config)
