"""
SmartFin Memory System — persistent user memory across sessions.

Stores financial data (transactions, goals, budgets) as Markdown files
with YAML frontmatter under ``.smartfin/memory/``.

Architecture:

    .smartfin/memory/
    ├── MEMORY.md                    # Index (YAML manifest of all files)
    ├── transactions/
    │   └── 2026-05.md               # Monthly transaction records
    ├── incomes/
    │   └── 2026-05.md
    ├── goals/
    │   └── emergency-fund.md        # One file per goal
    └── budgets/
        └── 2026-05-plan.md

Index file (MEMORY.md) is injected into the supervisor's system prompt.
A lightweight LLM scans the index + user message to decide which files
to load for the current turn.

Public API:
    - ``write_memory`` — persist agent outputs to memory files
    - ``retrieve_memory`` — recall relevant memory for the current request
"""

from app.memory.writer import write_memory
from app.memory.retriever import retrieve_memory

__all__ = ["write_memory", "retrieve_memory"]
