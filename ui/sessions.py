"""
Session + trace persistence for the web chat UI.

Tables (same schema as before, now in PostgreSQL):
  sessions(thread_id TEXT PRIMARY KEY, title TEXT, created_at FLOAT, last_activity_at FLOAT)
  traces(thread_id TEXT, position INTEGER, role TEXT, agent TEXT, content TEXT,
         checkpoint_id TEXT, PRIMARY KEY (thread_id, position))

The `sessions` table is the sidebar's source of truth. The `traces` table
is the flat log of chat bubbles keyed by thread_id and ordered by `position`.
`checkpoint_id` on a user-role row is the graph-state checkpoint id that
existed *before* that user message was submitted.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://smartfin:smartfin@localhost:5432/smartfin",
)

_pool: ConnectionPool | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    last_activity_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    thread_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    agent TEXT,
    content TEXT NOT NULL,
    checkpoint_id TEXT,
    extra TEXT,
    PRIMARY KEY (thread_id, position)
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
    ON sessions (last_activity_at DESC);
"""


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(_DB_URL, min_size=1, max_size=4, open=True, kwargs={"row_factory": dict_row})
    return _pool


def init_db() -> None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        conn.commit()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def auto_title(first_message: str, max_len: int = 48) -> str:
    text = (first_message or "").strip().replace("\n", " ")
    if not text:
        return "New chat"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def create_session(title: str | None = None) -> str:
    thread_id = f"ui-{uuid.uuid4().hex[:10]}"
    now = time.time()
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (thread_id, title, created_at, last_activity_at) "
                "VALUES (%s, %s, %s, %s)",
                (thread_id, title or "New chat", now, now),
            )
        conn.commit()
    return thread_id


def list_sessions() -> list[dict[str, Any]]:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, title, created_at, last_activity_at "
                "FROM sessions ORDER BY last_activity_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]


def get_session(thread_id: str) -> dict[str, Any] | None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, title, created_at, last_activity_at "
                "FROM sessions WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def rename_session(thread_id: str, new_title: str) -> None:
    title = (new_title or "").strip() or "Untitled chat"
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE sessions SET title = %s WHERE thread_id = %s", (title, thread_id))
        conn.commit()


def touch_session(thread_id: str) -> None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET last_activity_at = %s WHERE thread_id = %s",
                (time.time(), thread_id),
            )
        conn.commit()


def delete_session(thread_id: str) -> None:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM traces WHERE thread_id = %s", (thread_id,))
            cur.execute("DELETE FROM sessions WHERE thread_id = %s", (thread_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------

def load_trace(thread_id: str) -> list[dict[str, Any]]:
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, agent, content, checkpoint_id, extra "
                "FROM traces WHERE thread_id = %s ORDER BY position ASC",
                (thread_id,),
            )
            rows = cur.fetchall()

    trace: list[dict[str, Any]] = []
    for r in rows:
        entry: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["agent"]:
            entry["agent"] = r["agent"]
        if r["checkpoint_id"]:
            entry["checkpoint_id"] = r["checkpoint_id"]
        if r["extra"]:
            try:
                entry.update(json.loads(r["extra"]))
            except json.JSONDecodeError:
                pass
        trace.append(entry)
    return trace


def save_trace(thread_id: str, trace: list[dict[str, Any]]) -> None:
    """Replace the stored trace for this thread_id wholesale."""
    rows = []
    for i, entry in enumerate(trace):
        extras = {
            k: v
            for k, v in entry.items()
            if k not in {"role", "agent", "content", "checkpoint_id"}
        }
        rows.append(
            (
                thread_id,
                i,
                entry.get("role", ""),
                entry.get("agent"),
                entry.get("content", ""),
                entry.get("checkpoint_id"),
                json.dumps(extras) if extras else None,
            )
        )

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM traces WHERE thread_id = %s", (thread_id,))
            if rows:
                cur.executemany(
                    "INSERT INTO traces "
                    "(thread_id, position, role, agent, content, checkpoint_id, extra) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    rows,
                )
        conn.commit()


import logging
import time

logger = logging.getLogger(__name__)


def _init_db_with_retry(max_attempts: int = 5, backoff: float = 2.0) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            init_db()
            return
        except Exception as exc:
            if attempt < max_attempts:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(
                    "Failed to init UI session DB (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, max_attempts, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Failed to init UI session DB after %d attempts: %s",
                    max_attempts, exc,
                )
                raise


# Initialise on import so callers don't need to remember to.
_init_db_with_retry()
