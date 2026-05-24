"""
Redis cache for LLM responses — avoids redundant Anthropic API calls.

Public API:
    get_cached_llm_response(prompt_name, input_text) -> dict | None
    cache_llm_response(prompt_name, input_text, result_dict) -> None

Errors are logged but never raised — caching is best-effort.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
_TTL_SECONDS = 3600  # 1 hour

_redis: Any = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis
        _redis = redis.Redis.from_url(_REDIS_URL, socket_connect_timeout=2, decode_responses=True)
        _redis.ping()
        logger.debug("[cache] Redis connected to %s", _REDIS_URL)
    except Exception as exc:
        logger.warning("[cache] Redis unavailable, caching disabled: %s", exc)
        _redis = False
    return _redis


def _build_key(prompt_name: str, input_text: str) -> str:
    digest = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    return f"llm:{prompt_name}:{digest}"


def get_cached_llm_response(prompt_name: str, input_text: str) -> dict | None:
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(_build_key(prompt_name, input_text))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug("[cache] read miss: %s", exc)
        return None


def cache_llm_response(prompt_name: str, input_text: str, result: dict) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        r.set(_build_key(prompt_name, input_text), json.dumps(result, default=str), ex=_TTL_SECONDS)
        logger.debug("[cache] stored llm:%s", prompt_name)
    except Exception as exc:
        logger.debug("[cache] write skip: %s", exc)
