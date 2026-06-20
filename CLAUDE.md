# CLAUDE.md

## Project Overview
This is an agentic workflow for personal financial management, built with LangGraph (supervisor pattern), including 1 supervisor and 5 specialized agents where each is responsible for a specific task.

### Tech Stacks
- Anthropic Claude API - for LLM invocation
- Streamlit - for the chat interface
- FastAPI - for communications between the backend and frontend
- Docker Compose - for services containerizations
- PostgreSQL - for AppState checkpoint and session persistence
- Redis - for caching repetitive User input - LLM responses

---

## Key Commands

```bash
git pull --rebase
cp .env.example .env        # fill in ANTHROPIC_API_KEY, LANGCHAIN_API_KEY

# create venv
uv venv

# activate venv
# for Windows Powershell
.venv/Scripts/activate.ps1 
# for Windows Bash
.venv/Scripts/activate
# for Linux/MacOS
source .venv/bin/activate

# install dependencies
uv pip install -r requirements.txt

# start backend & frontend services
docker compose up -d

# run UI
streamlit run ui/app.py
```

---

## Key References Index

| Topic | Path |
| --- | --- |
| Environment variables | `@.env.example` |
| Coding Principles | `@.claude/rules/code-principle.md` |
| System Design Practices | `@.claude/rules/system-design.md`|
| Agent responsibilities and implementation notes | `@.claude/rules/agents.md` |
| Orchestrator entry point and public API | `@.claude/rules/orchestrator.md` |
| Test commands and coverage | `@.claude/rules/tests.md` |

<!-- Optional documents:
| Git workflow, branch naming, commit format | `CONTRIBUTING.md` |
| UI structure and page descriptions | `ui/README.md` |
-->

---

## Key Caveats

<!--- Run `git pull --rebase` to update from the remote repository
 - Never commit or push directly to `main`
- All changes must be branched off `dev`
- Branch type and commit type must match → see `CONTRIBUTING.md`-->
- **YOU MUST** activate the venv and run docker compose to start the backend and frontend services, and to run a test. This is important
- Follow `@.claude/rules/code-principle.md` during any code modification that include complex dependencies and multiple modules
- Follow `@.claude/rules/system-design.md` when designing or refactoring a service
- Use **Plan Mode** before any modification. Split the task. Show me the plan details with reasoning **By Subtask**. DO NOT execute directly, even when auto-accept is turned on
- Use Subagent-Driven Development at execution. Specify each subtask to a subagent. Run subagents in parallel or sequence judging by the dependencies between subtasks
- Always check the code syntax after every modification and do a validation yourself