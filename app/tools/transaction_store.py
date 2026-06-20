"""
Transaction store — file-based persistence for categorised transaction data.

Stores per-thread analysis results under .smartfin_cache/<thread_id>.json
so they survive process restarts and avoid re-running expensive LLM categorisation calls.

Public API:
    save_analysis(thread_id, categorised, trends) -> None
    load_analysis(thread_id) -> (list[Transaction], list[SpendingTrend]) | None
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.orchestrator.checkpoints import DATA_DIR
from app.state import SpendingTrend, Transaction

USER_CACHE_KEY = "user_default"

logger = logging.getLogger(__name__)

_CACHE_DIR = DATA_DIR.parent / ".smartfin_cache"


def _cache_path(thread_id: str) -> Path:
    """Return the JSON file path for a given thread_id. Sanitise to prevent path traversal."""
    safe_id = "".join(c for c in thread_id if c.isalnum() or c in "-_")
    return _CACHE_DIR / f"{safe_id}.json"


def save_analysis(
    thread_id: str,
    categorised: list[Transaction],
    trends: list[SpendingTrend],
) -> None:
    """
    Persist categorised transactions and spending trends to disk.

    Creates .smartfin_cache/ if it does not exist.
    Overwrites any existing file for this thread_id.
    Errors are logged but never raised — callers must not depend on this succeeding.
    """
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        path = _cache_path(thread_id)
        payload = {
            "categorised_transactions": [t.model_dump(mode="json") for t in categorised],
            "spending_trends": [s.model_dump(mode="json") for s in trends],
        }
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        logger.debug("[transaction_store] saved analysis for thread %s -> %s", thread_id, path)
    except Exception as exc:
        logger.error("[transaction_store] failed to save analysis for %s: %s", thread_id, exc)


def save_user_analysis(
    categorised: list[Transaction],
    trends: list[SpendingTrend],
) -> None:
    """Persist user-level (cross-session) categorised transactions and trends."""
    save_analysis(USER_CACHE_KEY, categorised, trends)


def load_user_analysis() -> tuple[list[Transaction], list[SpendingTrend]] | None:
    """Load user-level categorised transactions and trends, or None if not yet cached."""
    return load_analysis(USER_CACHE_KEY)


def load_analysis(
    thread_id: str,
) -> tuple[list[Transaction], list[SpendingTrend]] | None:
    """
    Load previously persisted categorised transactions and spending trends.

    Returns:
        (categorised_transactions, spending_trends) tuple if the cache file exists
        and parses cleanly, or None otherwise.
    """
    path = _cache_path(thread_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        categorised = [Transaction(**t) for t in raw.get("categorised_transactions", [])]
        trends = [SpendingTrend(**s) for s in raw.get("spending_trends", [])]
        logger.debug(
            "[transaction_store] loaded %d transactions, %d trends for thread %s",
            len(categorised),
            len(trends),
            thread_id,
        )
        return categorised, trends
    except Exception as exc:
        logger.error("[transaction_store] failed to load analysis for %s: %s", thread_id, exc)
        return None
