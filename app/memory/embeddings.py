"""OpenAI embedding client for SmartFin memory."""
from __future__ import annotations
import os
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed(text: str) -> list[float]:
    """Embed text using text-embedding-3-small. Raises on failure."""
    response = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call. Returns embeddings in the same order.

    Returns [] immediately if texts is empty. Raises on failure.
    """
    if not texts:
        return []
    response = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]
