"""
Memory retriever — recalls relevant user memories via vector similarity search.

Flow:
    1. Embed the user's message with text-embedding-3-small.
    2. Search smartfin_memory table for top-k similar records.
    3. Return {name, content} dicts for injection into the supervisor prompt.
    4. Any failure → return [] (non-blocking).
"""
from __future__ import annotations
import logging
from app.memory.embeddings import embed
from app.memory.vector_store import search

logger = logging.getLogger(__name__)


def retrieve_memory(user_message: str) -> list[dict[str, str]]:
    """Fetch memory records relevant to user_message via vector similarity."""
    try:
        query_vec = embed(user_message)
        rows = search(query_vec)
        return [{"name": row["id"], "content": row["content"]} for row in rows]
    except Exception as exc:
        logger.warning("[memory] Retrieval failed: %s", exc)
        return []
