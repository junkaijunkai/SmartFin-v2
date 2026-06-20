"""
SmartFin Memory System — persistent user memory across sessions.

Stores agent task summaries as natural-language text with vector embeddings
in the ``smartfin_memory`` PostgreSQL table (pgvector).

Architecture:

    PostgreSQL: smartfin_memory
    ├── id           — "goal_planning/emergency-fund", "budget_planning/2026-05"
    ├── memory_type  — agent name
    ├── content      — natural-language summary (embedded for similarity search)
    ├── description  — one-line summary for debug
    ├── embedding    — VECTOR(1536) from text-embedding-3-small
    └── updated_at

Public API:
    - ``write_task_memory`` — persist a completed agent task to the vector store
    - ``retrieve_memory``   — recall relevant memory via similarity search
"""

from app.memory.writer import write_task_memory
from app.memory.retriever import retrieve_memory

__all__ = ["write_task_memory", "retrieve_memory"]
