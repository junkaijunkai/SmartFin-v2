# SmartFin

> **An Agentic AI Co-Pilot for Personal Financial Management**

SmartFin is a multi-agent AI system that connects day-to-day spending behaviour with long-term financial wellbeing. Built on LangGraph's supervisor pattern, it coordinates five specialist agents that share a single typed state, communicate through tool calls, and pause for human confirmation before committing changes.

[中文文档](README.zh-CN.md)

---

## Features

| Feature | Description |
|---|---|
| **Expense Analysis** | Auto-categorises raw transactions and computes 30-day spending trends across categories |
| **Budget Planning** | Generates monthly budget allocations based on income and historical spending, evaluates progress, and warns before limits are breached |
| **Goal Tracking** | Extracts financial goals from natural language, calculates required monthly savings, and tracks progress |
| **Anomaly Detection** | Flags suspicious transactions using statistical methods (IQR, frequency analysis) with natural-language explanations |
| **Health Assessment** | Rates overall financial health via debt-to-income ratio, liquid reserve months, income concentration risk, and overspending indicators |
| **Human-in-the-Loop** | Pauses execution at critical decision points and presents a confirmation card; the user can approve, reject, or clarify before changes are committed |
| **Persistent Memory** | Remembers financial data across sessions — goals, budgets, and transaction history are stored as Markdown files and recalled when relevant |

---

## Architecture

SmartFin is built on a **LangGraph Supervisor** pattern. One orchestrator node classifies user intent and routes to specialist agents through a shared `AppState` TypedDict.

```
User Message
     │
     ▼
┌────────────────┐     active_agent     ┌──────────────────────┐
│  Memory Loader │─────────────────────▶│      Supervisor      │
│  (trace ctx +  │                      │  (intent classifier) │
│   memory recall)│                      └──────────┬───────────┘
└────────────────┘                                 │
                                                route_to_agent()
         ┌──────────────┬─────────────┬──────────────┼──────────────┬──────────────┐
         ▼              ▼             ▼              ▼              ▼              ▼
   Expense        Budget         Goal          Anomaly         Health
   Analysis       Planning       Planning      Detection       Assessment
         │              │             │              │              │
         └──────────────┴─────────────┴──────────────┴──────────────┘
                                              │
                                     pending_confirmation?
                                              │
                                              ▼
                                     ┌────────────────┐
                                     │  HITL Confirm  │ ← interupt_before
                                     │  (user review) │
                                     └────────────────┘
                                              │
                                              ▼
                                     Memory Saver
                                     (persist state)
```

### Graph Walkthrough

Every turn follows the same flow:

1. **Memory Loader** — Sets trace context and retrieves relevant long-term memory (goals, budgets from previous sessions) based on the user's latest message.
2. **Supervisor** — Classifies intent via LLM. When transaction data is new, it pre-pends `expense_analysis` as a data-preparation step and stores the user's real intent as `pending_intent`. After an agent finishes, the supervisor decides whether to route the next agent or end.
3. **Specialist Agent** — The routed agent runs its ReAct loop, calls deterministic tools, and writes results to the shared state.
4. **Memory Saver** — Persists state to a filesystem backup and writes long-term memory if the agent's output is confirmed (non-tentative).
5. **HITL Confirm** — If the agent requested confirmation, the graph pauses here via `interrupt_before`. The UI shows a confirmation card; the user's decision resumes the graph atomically.

### Shared State

All agents communicate through a single `AppState` TypedDict defined in `app/state.py`. Each agent declares which fields it can read and write in `app/orchestrator/state_view.py` — a runtime-enforced scope that prevents one agent from polluting another's data:

```
transactions ──▶ categorised_transactions ──▶ spending_trends
                                          ──▶ budget_allocations
                                          ──▶ anomaly_flags
                                          ──▶ health_summary
```

---

## Agent Design

Each specialist agent follows the **ReAct (Reasoning + Acting)** pattern: the LLM iteratively reasons about the user's request, calls deterministic tools, observes results, and produces a final answer.

### How Tool Calling Works

1. Each agent defines a set of `@tool`-decorated functions (using `langchain_core.tools`). Tools are pure Python — they parse parameters, call deterministic business logic, and return a string result.
2. The agent binds its tools to a `ChatAnthropic` instance via `.bind_tools()`.
3. `run_react_loop()` (`app/agents/react_utils.py`) manages the loop: it sends the system prompt + user message, invokes the LLM, and routes tool calls to the appropriate function.
4. The `final_answer` tool (shared by all agents) signals completion. Its structured arguments (`summary`, `needs_hitl_confirmation`, `hitl_summary`, `hitl_details`) are captured by the loop and written into `pending_confirmation` for HITL.

### Agent Tools

| Agent | Tools |
|---|---|
| **Expense Analysis** | `categorise_transactions_tool`, `compute_spending_trends_tool` |
| **Budget Planning** | `extract_budget_request_tool`, `generate_allocations_tool`, `calculate_spending_tool`, `evaluate_progress_tool`, `generate_warnings_tool`, `validate_budget_tool` |
| **Goal Planning** | `extract_goal_tool`, `create_goal_tool`, `calculate_required_saving_tool` |
| **Anomaly Detection** | `run_statistical_detection_tool`, `generate_explanation_tool` |
| **Health Assessment** | `compute_health_assessment_tool` |

---

## Human-in-the-Loop

Critical financial decisions pause for user confirmation:

1. The agent sets `needs_hitl_confirmation=True` in its `final_answer` call. The graph is compiled with `interrupt_before=["confirm"]`, so execution pauses before the `confirm` node.
2. The UI receives a `__pause__` SSE event containing the `pending_confirmation` payload (action type, summary, details).
3. The user approves, rejects, or provides clarifying text.
4. The UI sends a single `POST /threads/{id}/runs/stream` with a `resume` payload. The graph resumes atomically — no separate PATCH request, eliminating the race condition between resuming and reading state.

---

## Memory System

SmartFin persists financial data as **Markdown files with YAML frontmatter** under `.smartfin/memory/`:

```
.smartfin/memory/
├── MEMORY.md                    # Index file
├── transactions/
│   └── 2026-05.md               # Monthly transaction records
├── incomes/
│   └── 2026-05.md
├── goals/
│   └── emergency-fund.md        # One file per goal
└── budgets/
    └── 2026-05-plan.md
```

At the start of each turn, a lightweight LLM (with a 2-second timeout) scans the index and selects files relevant to the user's message. Their content is injected into the agent's system prompt as user history context. The system degrades gracefully — if the LLM call fails or times out, no memory is loaded.

---

## Observability

Every request produces a structured trace with the following hierarchy:

```
trace_id (per request)
  └── span_id (per graph node)
       └── parent_span_id (links child to parent)
            └── events: STATE_SNAPSHOT, TOOL_CALL, TOKEN_USAGE, API_REQUEST, ...
```

Traces are written as JSONL files under `.smartfin/traces/`, one file per `thread_id`. Token usage is captured via a LangChain callback handler. Error events are categorised (validation error, tool error, LLM error, internal error) for programmatic filtering.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (recommended) or local Python environment
- An [Anthropic API key](https://console.anthropic.com/)

### Run with Docker

```bash
git clone https://github.com/junkaijunkai/SmartFin-v2.git
cd SmartFin-v2

# Configure environment
cp .env.example .env
# → Fill in ANTHROPIC_API_KEY

# Start all services
docker compose up --build
```

- **Streamlit UI**: `http://localhost:8501`
- **Backend API**: `http://localhost:8000`

### Run Locally

```bash
pip install -r requirements.txt
cp .env.example .env  # → add ANTHROPIC_API_KEY

# Start the backend
uvicorn app.api:app --host 0.0.0.0 --port 8000

# In another terminal, start the UI
streamlit run ui/app.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) (Supervisor pattern) |
| **LLM** | [Anthropic Claude](https://www.anthropic.com/) (Haiku by default, Sonnet configurable) |
| **LLM Framework** | [LangChain](https://www.langchain.com/) |
| **Backend** | FastAPI |
| **UI** | Streamlit |
| **Data Validation** | Pydantic v2 |
| **Observability** | LangSmith, structured JSONL logging |
| **Infrastructure** | Docker Compose, PostgreSQL (state checkpointing), Redis (caching) |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API access |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `smartfin`) |
| `SMARTFIN_MODEL` | No | Claude model ID or alias (default: `claude-haiku-4-5`) |
| `SMARTFIN_ENFORCE_APPROVED_MODELS` | No | When `true`, unapproved model IDs fall back to the registry default |
| `SMARTFIN_LOG_FORMAT` | No | `plain` or `json` logging output |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/analyze` | One-shot analysis (non-streaming, returns structured response) |
| `POST` | `/threads/{id}/runs/stream` | Streaming execution with SSE, supports HITL resume via `resume` payload |
| `GET` | `/threads/{id}/state` | Read current thread state (for debugging) |

---

## License

This project is developed for academic purposes.
