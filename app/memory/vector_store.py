"""PostgreSQL + pgvector backed memory store for SmartFin."""
from __future__ import annotations

import logging
import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://smartfin:smartfin@localhost:5432/smartfin",
)

# Module-level flag — prevents CREATE TABLE from running more than once per process
_table_ready: bool = False

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS smartfin_memory (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    description TEXT,
    embedding VECTOR(1536),
    updated_at TIMESTAMPTZ DEFAULT NOW()
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS smartfin_memory_embedding_idx
ON smartfin_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)
"""

_UPSERT_SQL = """
INSERT INTO smartfin_memory (id, memory_type, content, description, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    memory_type = EXCLUDED.memory_type,
    content     = EXCLUDED.content,
    description = EXCLUDED.description,
    embedding   = EXCLUDED.embedding,
    updated_at  = NOW();
"""

_SEARCH_SQL = """
SELECT id, memory_type, content, description,
       1 - (embedding <=> %s::vector) AS similarity
FROM smartfin_memory
WHERE 1 - (embedding <=> %s::vector) > %s
ORDER BY similarity DESC, updated_at DESC
LIMIT %s
"""


def _ensure_table() -> None:
    """Create the smartfin_memory table and index if they do not exist.

    Idempotent: SQL is only sent once per process lifetime via ``_table_ready``.
    CREATE EXTENSION must run before register_vector (which queries the type OID).
    """
    global _table_ready
    if _table_ready:
        return

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)

    _table_ready = True


def upsert(
    id: str,
    memory_type: str,
    content: str,
    description: str,
    embedding: list[float],
) -> None:
    """Insert or update a memory record."""
    _ensure_table()

    with psycopg.connect(_DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_UPSERT_SQL, (id, memory_type, content, description, embedding))

    logger.debug("[vector_store] Upserted memory id=%s", id)


def search(
    query_embedding: list[float],
    top_k: int = 5,
    similarity_threshold: float = 0.35,
) -> list[dict]:
    """Return the top-k memories whose cosine similarity exceeds the threshold."""
    _ensure_table()

    with psycopg.connect(_DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                _SEARCH_SQL,
                (query_embedding, query_embedding, similarity_threshold, top_k),
            )
            rows = cur.fetchall()

    logger.debug("[vector_store] Search returned %d rows", len(rows))
    return list(rows)
