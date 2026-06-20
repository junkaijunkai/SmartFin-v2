# Orchestrator: LangGraph Supervisor Pattern

The orchestrator module wires together all specialist agents under a single supervisor node and manages their execution flow, state persistence, and Human-in-the-Loop (HITL) confirmation.

## Architecture Overview

```
              ENTRY: memory_loader
                       │
                 SUPERVISOR NODE
            (LLM-based intent classification)
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
 expense_analysis  budget_planning  goal_planning  ...
      │                │                │
      └────────────────┼────────────────┘
                       │
                 memory_saver
                  /           \
        CONFIRM NODE        memory_loader
        (HITL pause)        (next turn / no HITL)
              │
        memory_loader
```

## Key Concepts

### memory_loader Node
Entry point for every turn. Retrieves relevant long-term memory based on the latest user message and sets trace context. The LangGraph checkpoint is the source of truth for all other state fields.

### Supervisor Node
- Reads user intent from the latest message
- Input is first passed through `filter_user_input()` to block prompt-injection attempts
- Routes to the appropriate specialist agent based on classified intent
- Handles loop continuation: when `expense_analysis` must run as a prerequisite, stores the original target in `pending_intent` and re-routes after prep completes
- Responds with a scoped error message for `unknown` intents

**Intent classification** is handled by `intent_classifier.py`, which uses `ChatAnthropic` with structured output (`_IntentResult`) for LLM-based classification. Keyword matching (`_keyword_fallback()`) is a secondary safety net applied only when: (a) the LLM call fails/times out, or (b) the LLM returns `unknown` and keyword matching can produce a confident override.

### Agent Nodes
Each specialist agent (expense_analysis, budget_planning, etc.) is a LangGraph node entry point that:
- Receives an `AgentStateView` — a scoped projection of `AppState` — not the full state object. Every agent node is wrapped by `_wrap_agent()` in `graph.py`, which enforces field-level read/write permissions declared in `AGENT_SCOPES` (`state_view.py`). Reading an undeclared field raises `KeyError`; writing one raises `ValueError`.
- Performs analysis (via ReAct loop or direct sequential call — see `agents.md`)
- Writes results back to `AppState` via the scoped view
- Optionally sets `pending_confirmation` to trigger HITL review

### memory_saver Node
Runs after every specialist agent. Persists structured state to the filesystem via `MemoryStore`. Skips `save_state()` when `pending_confirmation.action` starts with `"approve_"` (tentative writes await HITL approval). Also writes confirmed agent outputs to long-term memory.

### HITL Confirmation
The graph is compiled with `interrupt_before=[NODE_CONFIRM]`, pausing execution before the confirm node runs. The flow to reach HITL is:

```
agent node → memory_saver → CONFIRM NODE (paused here)
```

The UI layer:
1. Reads `state["pending_confirmation"]` payload
2. Presents it to the user for review
3. Calls `resume_with_confirmation(...)` to continue

`resume_with_confirmation` writes the user's decision into a separate `hitl_decision` field (not into `pending_confirmation`) to preserve correct routing through `route_after_agent`.

### Checkpointer
Persists full `AppState` after every node execution via `PostgresSaver`. The connection targets `DATABASE_URL` (defaults to `postgresql://smartfin:smartfin@localhost:5432/smartfin`) and is initialised with retry/backoff logic in `_init_checkpointer()`. State model types are pre-registered with `JsonPlusSerializer` to suppress deserialisation warnings.

## Public API

### Core Graph Object

```python
from app.orchestrator import app_graph

config = {"configurable": {"thread_id": "user-123"}}
result = app_graph.invoke(
    {"messages": [HumanMessage(content="Analyze my spending")]}, 
    config
)
```

### HITL Helpers

```python
from app.orchestrator import get_pending_interrupt, resume_with_confirmation

# Check if graph is paused at interrupt
state = get_pending_interrupt(app_graph, config)
if state:
    print("Pending confirmation:", state["pending_confirmation"])
    
    # User reviews and decides
    resume_with_confirmation(app_graph, config, confirmed=True)
```

## Module Files

| File | Responsibility |
|------|-----------------|
| `graph.py` | StateGraph definition, node wiring, `_wrap_agent` scoping wrapper, compilation |
| `router.py` | Node name constants; `route_to_agent` (supervisor → agent) and `route_after_agent` (memory_saver → confirm or memory_loader) |
| `checkpoints.py` | PostgresSaver setup with retry/backoff, HITL helpers (`get_pending_interrupt`, `resume_with_confirmation`) |
| `intent_classifier.py` | LLM-based intent classification (`ChatAnthropic` + structured output) with keyword fallback |
| `state_view.py` | `AgentStateView` and `AGENT_SCOPES` — field-level read/write scoping for each agent |
| `context.py` | Message compression and conversation history summarisation (triggers at >30 messages or >60k estimated tokens) |
| `__init__.py` | Public API exports (`app_graph`, `get_pending_interrupt`, `resume_with_confirmation`) |

## Extending the Orchestrator

To add a new specialist agent:

1. Create the agent module (e.g., `app/agents/new_agent/agent.py`) with a `run(view: AgentStateView) -> dict` function
2. Add the agent's field scope to `AGENT_SCOPES` in `state_view.py`
3. Add a node function in `graph.py`:
   ```python
   def new_agent_node(state: AppState) -> dict:
       from app.agents.new_agent.agent import run
       return run(state)
   ```
4. Register the node in `build_graph()` using `_wrap_agent`:
   ```python
   builder.add_node(NODE_NEW_AGENT, _wrap_agent(NODE_NEW_AGENT, new_agent_node))
   ```
5. Add an edge from the new node to `NODE_MEMORY_SAVER`
6. Update `route_to_agent` routing map in `router.py` to include the new agent
7. Export the updated `app_graph` from `__init__.py`
