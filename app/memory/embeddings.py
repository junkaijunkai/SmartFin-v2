"""OpenAI embedding client for SmartFin memory, routed through LiteLLM gateway."""
from __future__ import annotations
import os
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # Route through the LiteLLM gateway so OPENAI_API_KEY stays in the
        # gateway container only. The gateway maps text-embedding-3-small →
        # openai/text-embedding-3-small using its own OPENAI_API_KEY.
        _client = OpenAI(
            base_url=os.getenv("LITELLM_BASE_URL", "http://gateway:4000/v1"),
            api_key=os.getenv("LITELLM_VIRTUAL_KEY", "placeholder"),
        )
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
