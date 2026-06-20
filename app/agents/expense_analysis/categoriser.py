"""
Transaction categoriser — uses Claude to classify raw transactions in chunked
batch calls, returning the same list with the category field populated.

Chunking:     large inputs are split into chunks of CHUNK_SIZE before sending
              to the LLM, then results are merged back in original order.
Retry policy: each chunk is retried up to MAX_RETRIES times with exponential
              backoff before degrading.
Degradation:  if all retries for a chunk fail, that chunk falls back to
              keyword-based rule matching so downstream agents still receive
              partial categorisation.
"""

from __future__ import annotations

import logging
import os
import re
import time

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from app.config import resolve_model_name, get_prompt
from app.state import Transaction, TransactionCategory

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
CHUNK_SIZE = 200        # transactions per LLM call
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each subsequent retry
_LLM_TIMEOUT = 30       # seconds


# ---------------------------------------------------------------------------
# Structured-output schema (internal, not exported)
# ---------------------------------------------------------------------------


class _CategoryResult(BaseModel):
    transaction_id: str
    category: TransactionCategory


class _CategoryBatch(BaseModel):
    results: list[_CategoryResult]


# ---------------------------------------------------------------------------
# Keyword fallback (used when all LLM retries for a chunk are exhausted)
# ---------------------------------------------------------------------------

# Each entry: (list of lowercase keywords, category to assign on any match).
# Rules are evaluated in order; first match wins.
# Input text is sanitised and matched on word boundaries before comparison —
# see _keyword_fallback() for details.
_KEYWORD_RULES: list[tuple[list[str], TransactionCategory]] = [
    (["salary", "payroll", "wage", "dividend", "employer"], TransactionCategory.INCOME),
    (["rent", "mortgage", "landlord"], TransactionCategory.HOUSING),
    (["tesco", "sainsbury", "waitrose", "aldi", "lidl", "grocery",
      "supermarket", "restaurant", "cafe", "coffee", "nando", "pret",
      "takeaway", "food"], TransactionCategory.FOOD),
    (["uber", "lyft", "taxi", "tfl", "train", "bus", "transport",
      "rail", "tube", "metro"], TransactionCategory.TRANSPORT),
    (["netflix", "spotify", "cinema", "vue", "odeon", "theatre",
      "entertainment", "game", "steam"], TransactionCategory.ENTERTAINMENT),
    (["electric", "electricity", "gas", "water", "internet", "bt",
      "broadband", "utility", "utilities"], TransactionCategory.UTILITIES),
    (["gym", "doctor", "gp", "pharmacy", "hospital", "health",
      "dental", "boots", "prescription"], TransactionCategory.HEALTHCARE),
    (["course", "udemy", "coursera", "university", "school",
      "education", "tuition", "textbook"], TransactionCategory.EDUCATION),
    (["amazon", "nike", "adidas", "h&m", "zara", "clarks",
      "shop", "store", "mall", "clothing", "clothes"], TransactionCategory.SHOPPING),
    (["savings", "isa", "investment", "deposit", "pension"], TransactionCategory.SAVINGS),
]


def _sanitise(text: str, max_len: int = 60) -> str:
    """
    Normalise text before keyword matching to reduce injection surface.
    Strips to lowercase alphanumeric + spaces, then truncates.
    """
    cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return cleaned[:max_len]


def _keyword_fallback(transactions: list[Transaction]) -> list[Transaction]:
    """
    Best-effort rule-based categorisation used when the LLM is unavailable.

    Input text is sanitised and matched on word boundaries before comparison.
    Negative amounts are always treated as income.
    Falls back to OTHER when no keyword matches.
    """
    result: list[Transaction] = []
    for t in transactions:
        if t.amount < 0:
            category = TransactionCategory.INCOME
        else:
            text = _sanitise(f"{t.description} {t.merchant}")
            category = TransactionCategory.OTHER
            for keywords, cat in _KEYWORD_RULES:
                if any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords):
                    category = cat
                    break
        result.append(t.model_copy(update={"category": category}))
    return result


# ---------------------------------------------------------------------------
# Per-chunk LLM call with retry
# ---------------------------------------------------------------------------




def _categorise_chunk(
    chunk: list[Transaction],
    structured_llm,
) -> tuple[list[Transaction], bool]:
    """
    Classify one chunk of transactions via the LLM with retry + fallback.

    Returns:
        (categorised transactions, llm_succeeded)
        llm_succeeded is False when the keyword fallback was used.
    """
    category_values = ", ".join(c.value for c in TransactionCategory)
    lines = [
        f"id={t.id} | merchant={t.merchant} | description={t.description} | amount={t.amount:.2f}"
        for t in chunk
    ]
    messages = get_prompt("expense_categoriser").format_messages(
        category_values=category_values,
        transactions_text="\n".join(lines),
    )
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response: _CategoryBatch = structured_llm.invoke(messages)
            category_map = {r.transaction_id: r.category for r in response.results}
            categorised = [
                t.model_copy(update={"category": category_map.get(t.id, TransactionCategory.OTHER)})
                for t in chunk
            ]
            return categorised, True
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = _INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.warning(
                    "LLM categorisation failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM categorisation failed after %d attempts: %s",
                    MAX_RETRIES, last_exc,
                )

    logger.warning("Falling back to keyword-based categorisation for this chunk.")
    return _keyword_fallback(chunk), False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def categorise_transactions(
    transactions: list[Transaction],
) -> tuple[list[Transaction], bool]:
    """
    Assign a TransactionCategory to every transaction.

    Large inputs are split into chunks of CHUNK_SIZE and processed
    sequentially. Results are merged back in the original input order.

    Returns:
        (categorised transactions, llm_succeeded)
        llm_succeeded is True only when every chunk was classified by the LLM.
        If any chunk fell back to keyword rules, llm_succeeded is False.
    """
    if not transactions:
        return [], True

    model_name = resolve_model_name("default")
    llm = ChatAnthropic(model=model_name, timeout=_LLM_TIMEOUT)
    structured_llm = llm.with_structured_output(_CategoryBatch)

    chunks = [
        transactions[i: i + CHUNK_SIZE]
        for i in range(0, len(transactions), CHUNK_SIZE)
    ]

    all_results: list[Transaction] = []
    chunk_ok_flags: list[bool] = []

    for idx, chunk in enumerate(chunks):
        logger.debug(
            "Categorising chunk %d/%d (%d transactions)", idx + 1, len(chunks), len(chunk)
        )
        categorised, ok = _categorise_chunk(chunk, structured_llm)
        all_results.extend(categorised)
        chunk_ok_flags.append(ok)

    llm_succeeded = all(chunk_ok_flags)
    if not llm_succeeded:
        failed = chunk_ok_flags.count(False)
        logger.warning("%d/%d chunk(s) used keyword fallback.", failed, len(chunks))

    return all_results, llm_succeeded
