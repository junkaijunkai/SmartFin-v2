"""
Memory retriever — recalls relevant user memories for the current request.

Flow:
    1. Read ``MEMORY.md`` index → list of available memory files.
    2. Send index + user message to a lightweight LLM → returns relevant file list.
    3. Read the selected files and return their content.
    4. Timeout / failure → return empty (non-blocking).

The returned content is injected into the supervisor's system prompt before
the main graph executes, so agents see relevant user history.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEMORY_ROOT = _REPO_ROOT / ".smartfin" / "memory"
_INDEX_FILE = "MEMORY.md"

# Timeout for the LLM selection call (seconds)
_RETRIEVAL_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_memory(user_message: str) -> list[dict[str, str]]:
    """Fetch memory files relevant to ``user_message``.

    Returns a list of ``{"name": str, "content": str}`` dicts, or an empty
    list if no index exists, the LLM call fails, or the timeout is reached.

    This function blocks for at most ``_RETRIEVAL_TIMEOUT_S`` seconds.
    """
    index = _load_index()
    if not index:
        return []

    selected = _select_files(index, user_message)
    if not selected:
        return []

    return _load_contents(selected)


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------


def _load_index() -> list[dict[str, str]]:
    """Parse MEMORY.md and return the file manifest.

    MEMORY.md format (one YAML list item per line)::

        - name: transactions/2026-05.md  description: "..."  type: transaction
    """
    index_path = _MEMORY_ROOT / _INDEX_FILE
    if not index_path.exists():
        return []

    entries: list[dict[str, str]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("- name:"):
            continue
        entry = _parse_index_line(line)
        if entry:
            entries.append(entry)
    return entries


def _parse_index_line(line: str) -> dict[str, str] | None:
    """Parse ``- name: path  description: "..."  type: t`` into a dict."""
    try:
        # Simple key: value parser
        parts = line.lstrip("- ").split("  ")
        result = {}
        for part in parts:
            if ":" in part:
                k, v = part.split(":", 1)
                result[k.strip()] = v.strip().strip('"').strip("'")
        if "name" in result and "type" in result:
            return result
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM file selection (lightweight model)
# ---------------------------------------------------------------------------


def _select_files(index: list[dict[str, str]], user_message: str) -> list[str]:
    """Ask a lightweight LLM which memory files are relevant.

    Falls back to returning all files on error or timeout.
    """
    # Build a compact file list for the LLM
    file_list = "\n".join(
        f"  [{i}] {e['name']} — {e.get('description', '')}"
        for i, e in enumerate(index)
    )

    prompt = (
        "You are a memory retrieval system for a personal finance assistant.\n"
        "Given the user's current message and a list of available memory files,\n"
        "return the indices of files that contain information relevant to the\n"
        "user's request.\n\n"
        "Examples:\n"
        "  User: 'How is my emergency fund progressing?' → [2]\n"
        "  User: 'Help me set a budget' → [0, 3]\n"
        "  User: 'Hi' → []\n\n"
        f"Available files:\n{file_list}\n\n"
        f"User message: {user_message}\n\n"
        "Return only a JSON array of indices, e.g. [0, 2] or []. No explanation."
    )

    try:
        import concurrent.futures
        from langchain_anthropic import ChatAnthropic
        from app.config import resolve_model_name

        model_name = resolve_model_name(os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5"))
        llm = ChatAnthropic(model=model_name, timeout=3)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(llm.invoke, prompt)
            response = future.result(timeout=_RETRIEVAL_TIMEOUT_S)

        raw = response.content.strip() if response.content else "[]"
        indices = json.loads(raw)
        if isinstance(indices, list):
            selected = [index[i]["name"] for i in indices if 0 <= i < len(index)]
            logger.info("[memory] Selected %d/%d files for recall", len(selected), len(index))
            return selected
        return []
    except concurrent.futures.TimeoutError:
        logger.warning("[memory] Retrieval timed out after %ss", _RETRIEVAL_TIMEOUT_S)
        return []
    except Exception as exc:
        logger.warning("[memory] Retrieval failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _load_contents(names: list[str]) -> list[dict[str, str]]:
    """Read the full content of each file, returning ``{name, content}``."""
    results: list[dict[str, str]] = []
    for name in names:
        path = _MEMORY_ROOT / name
        if not path.exists():
            logger.debug("[memory] File not found: %s", name)
            continue
        try:
            content = path.read_text(encoding="utf-8")
            results.append({"name": name, "content": content})
        except Exception as exc:
            logger.warning("[memory] Failed to read %s: %s", name, exc)
    return results
