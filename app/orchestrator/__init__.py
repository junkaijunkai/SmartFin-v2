"""
app.orchestrator

Public API for the orchestrator package.  Other parts of the codebase
(UI, tests, CLI) should import from here rather than from the submodules
directly, so internal structure can change without breaking call sites.

Usage:
    from app.orchestrator import app_graph, get_pending_interrupt, resume_with_confirmation

    config = {"configurable": {"thread_id": "user-123"}}

    # Start a new conversation turn
    result = app_graph.invoke({"messages": [HumanMessage(content="Analyse my spending")]}, config)

    # Check if graph is paused at a HITL interrupt
    state = get_pending_interrupt(app_graph, config)
    if state:
        resume_with_confirmation(app_graph, config, confirmed=True)
"""

from app.orchestrator.graph import build_graph
from app.orchestrator.checkpoints import (
    get_pending_interrupt,
    resume_with_confirmation,
)

# Build once at import time — the compiled graph is stateless itself;
# per-user state is stored in the checkpointer keyed by thread_id.
app_graph = build_graph()

__all__ = [
    "app_graph",
    "get_pending_interrupt",
    "resume_with_confirmation",
]
