# Agent Responsibilities

## Specialist Agents (`app/agents/`)

Each agent has an `agent.py` (LangGraph node entry point) and one or more logic modules. All agents are fully implemented.

### State Access
Agent `run()` functions receive an `AgentStateView` — a scoped projection of `AppState` from `app/orchestrator/state_view.py` — not the full state object. Attempting to read or write fields outside the agent's declared scope in `AGENT_SCOPES` raises `KeyError` / `ValueError` at runtime.

### Architecture Patterns
Two implementation patterns are used, chosen based on task structure:

- **ReAct + tool calling** — `expense_analysis`, `budget_planning`, `goal_planning`: The LLM drives an iterative loop, calls tools to perform sub-tasks, and produces a structured final answer. Applied where the task requires open-ended reasoning or dynamic decision-making.
- **Direct sequential** — `anomaly_detection`, `health_assessment`: Fixed deterministic pipeline with no ReAct overhead. Applied where the execution chain is concrete and fully predefined.

---

## Agent Directory Structure

### `expense_analysis/`
Transaction categorisation and trend analysis. Feeds downstream agents that depend on `categorised_transactions`.

| File | Role |
|------|------|
| `agent.py` | ReAct node entry — orchestrates categorisation and trend computation |
| `categoriser.py` | LLM-based transaction category assignment |
| `analyser.py` | 30-day spending trend computation |
| `extractor.py` | Extracts transactions from free-form user messages |

### `budget_planning/`
Budget allocation and overspend warnings.

| File | Role |
|------|------|
| `agent.py` | ReAct node entry — extracts budget intent, generates allocations, evaluates progress |
| `extractor.py` | Parses natural-language budget requests into structured parameters |
| `planner.py` | Generates allocations, calculates spending, evaluates progress, generates warnings |

### `goal_planning/`
Financial goal tracking and savings planning.

| File | Role |
|------|------|
| `agent.py` | ReAct node entry — extracts goal data, validates, creates goals, computes required savings |
| `extractor.py` | LLM-based goal extraction from free-form text (amount, target date, name) |
| `tracker.py` | Deterministic financial calculations (required monthly saving, progress %) |

### `anomaly_detection/`
Suspicious and unusual transaction detection.

| File | Role |
|------|------|
| `agent.py` | Sequential node entry — runs detection then explanation |
| `detector.py` | IQR and frequency-based statistical anomaly detection |
| `extractor.py` | LLM-based explanation generation for detected anomalies (with fallback) |

### `health_assessment/`
Debt-to-income ratio, reserve months, and risk rating.

| File | Role |
|------|------|
| `agent.py` | Sequential node entry — calls `assess_health()` and formats the response |
| `assessor.py` | Deterministic health metric computation with LLM advisory generation (silent fallback) |
