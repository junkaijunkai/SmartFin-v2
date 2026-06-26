"""PostgreSQL + pgvector backed memory store for SmartFin."""
from __future__ import annotations

import logging
import os
from datetime import datetime

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
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NULL
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS smartfin_memory_embedding_idx
ON smartfin_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)
"""

_MIGRATE_EXPIRES_AT_SQL = """
ALTER TABLE smartfin_memory ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ DEFAULT NULL
"""

_MIGRATE_HASH_SQL = """
ALTER TABLE smartfin_memory ADD COLUMN IF NOT EXISTS content_hash TEXT DEFAULT NULL
"""

_UPSERT_SQL = """
INSERT INTO smartfin_memory (id, memory_type, content, description, embedding, expires_at, content_hash)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    memory_type  = EXCLUDED.memory_type,
    content      = EXCLUDED.content,
    description  = EXCLUDED.description,
    embedding    = EXCLUDED.embedding,
    expires_at   = EXCLUDED.expires_at,
    content_hash = EXCLUDED.content_hash,
    updated_at   = NOW();
"""

_FETCH_HASHES_BATCH_SQL = """
SELECT id, content_hash FROM smartfin_memory WHERE id = ANY(%s)
"""

_SEARCH_SQL = """
SELECT id, memory_type, content, description,
       1 - (embedding <=> %s::vector) AS similarity
FROM smartfin_memory
WHERE (expires_at IS NULL OR expires_at > NOW())
  AND 1 - (embedding <=> %s::vector) > %s
ORDER BY similarity DESC, updated_at DESC
LIMIT %s
"""

_FETCH_CONTENT_SQL = """
SELECT content FROM smartfin_memory WHERE id = %s
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
        conn.execute(_MIGRATE_EXPIRES_AT_SQL)
        conn.execute(_MIGRATE_HASH_SQL)

    _table_ready = True


def fetch_content(id: str) -> str | None:
    """Return the stored content for a record, or None if not found.

    Kept for backward compatibility. Prefer fetch_hashes_batch for bulk change detection.
    """
    _ensure_table()

    with psycopg.connect(_DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_FETCH_CONTENT_SQL, (id,))
            row = cur.fetchone()

    return row["content"] if row else None


def fetch_hashes_batch(ids: list[str]) -> dict[str, str | None]:
    """Return stored content_hash values for the given record ids in one query.

    Returns a dict mapping id → content_hash (None if the record has no hash yet).
    Ids not present in the table are absent from the returned dict.
    """
    if not ids:
        return {}

    _ensure_table()

    with psycopg.connect(_DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_FETCH_HASHES_BATCH_SQL, (ids,))
            rows = cur.fetchall()

    return {row["id"]: row["content_hash"] for row in rows}


def upsert(
    id: str,
    memory_type: str,
    content: str,
    description: str,
    embedding: list[float],
    expires_at: datetime | None = None,
    content_hash: str | None = None,
) -> None:
    """Insert or update a memory record."""
    _ensure_table()

    with psycopg.connect(_DATABASE_URL, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(_UPSERT_SQL, (id, memory_type, content, description, embedding, expires_at, content_hash))

    logger.debug("[vector_store] Upserted memory id=%s", id)


def batch_upsert(records: list[dict]) -> None:
    """Insert or update multiple memory records in a single transaction.

    Each dict must have keys: id, memory_type, content, description, embedding, expires_at.
    Optional key: content_hash (SHA-256 hex digest of content, for deduplication).
    """
    if not records:
        return

    _ensure_table()

    params = [
        (r["id"], r["memory_type"], r["content"], r["description"], r["embedding"], r.get("expires_at"), r.get("content_hash"))
        for r in records
    ]

    with psycopg.connect(_DATABASE_URL, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(_UPSERT_SQL, params)

    logger.debug("[vector_store] Batch upserted %d memory records", len(records))


def search(
    query_embedding: list[float],
    top_k: int = 5,
    similarity_threshold: float = 0.55,
) -> list[dict]:
    """Return the top-k non-expired memories whose cosine similarity exceeds the threshold."""
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
