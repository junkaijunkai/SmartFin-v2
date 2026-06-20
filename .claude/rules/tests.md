---
paths: 
  - "tests/**/*.py"
---
# Test Commands and Coverage

## Setup

Always activate the virtual environment:

```bash
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

## Running Tests

### Full Test Suite
```bash
pytest tests/ -v
```

Runs all unit, integration, and security tests with coverage report (auto-configured in `pyproject.toml`).

### Unit Tests Only
```bash
pytest tests/unit/ -v
```

Fast, isolated tests for individual agent logic and utilities. No external API calls or graph invocation.

**Coverage by agent:**
- `test_expense_analysis.py` — Transaction categorisation + trend analysis
- `test_anomaly_detection.py` — Transaction anomaly detection
- `test_budget_planning.py` — Budget allocation
- `test_goal_planning.py` — Savings Goal tracking
- `test_health_assessment.py` — Financial health assessment

### Integration Tests
```bash
python -m pytest tests/integration/test_agent_pipeline.py -v
```

Smoke tests verifying the orchestrator graph compiles, routes correctly, and completes without errors. Tests run against stub agent implementations, no real LLM calls.

**Tests:**
- Graph compilation
- Supervisor routing logic
- Agent queue execution
- HITL interrupt behavior

### Security Tests
```bash
python -m pytest tests/security/ -v
```

Guardrails and safety validation (e.g., prompt injection, data leakage).

### LLMSecOps Policy Checks
```bash
python scripts/llmsecops_ci.py
```

Runs deterministic CI policy checks for approved model registry usage, prompt injection blocking, and sensitive output redaction.

### Single Test
```bash
python -m pytest tests/integration/test_agent_pipeline.py::test_graph_compiles -v
```

Run a specific test by name.

## Coverage

Coverage is automatically collected for the `app/` module (see `pyproject.toml`). View the report after running tests:

```bash
python -m pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html  # macOS
```

**Current coverage goal:** Maintain >80% for core agent logic; 100% for orchestrator.

## Test Fixtures

Common test data lives in `tests/fixtures/`:
- `sample_transactions.json` — Real-world transaction samples
- `generate_transactions.py` — Utilities to generate synthetic transaction data

Usage:
```python
from tests.fixtures.sample_transactions import SAMPLE_TRANSACTIONS
```

## Mocking Strategy

### Unit Tests
- Use `unittest.mock` to patch external services (e.g., `ChatAnthropic` for LLM calls)
- Keep mocks simple and focused on the agent's logic, not the service

### Integration Tests
- No mocks; run against real (stub) orchestrator and checkpointer
- Use in-memory MemorySaver, not a real database

### Real API Calls
- Never make real API calls in tests
- Use environment variable guards or mock services
- Add `@pytest.mark.skip_if_no_api` for tests that require credentials
