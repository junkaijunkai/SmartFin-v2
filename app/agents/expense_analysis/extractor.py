from __future__ import annotations

import os
import re
import uuid
from datetime import datetime

from pydantic import BaseModel, Field

_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_TRANSACTION_VERBS = frozenset([
    "spent", "paid", "bought", "purchased", "cost", "charged", "spend",
])


class _SingleTxn(BaseModel):
    amount: float = Field(description="Transaction amount as a positive number")
    merchant: str | None = Field(default=None, description="Merchant or payee name")
    description: str | None = Field(default=None, description="Short description of what was purchased")
    category: str | None = Field(
        default=None,
        description="One of: food, transport, housing, entertainment, healthcare, education, shopping, utilities, income, savings, other",
    )


class _TxnExtract(BaseModel):
    transactions: list[_SingleTxn] = Field(
        default_factory=list,
        description="All transactions explicitly described in the message. Empty list if none found.",
    )


def _quick_filter(message: str) -> bool:
    """Cheap pre-check — skip LLM call if no numeric amount + transaction verb found."""
    lower = message.lower()
    return bool(_AMOUNT_RE.search(message)) and any(v in lower for v in _TRANSACTION_VERBS)


def extract_transaction_from_message(
    message: str,
    current_date: str | None = None,
) -> "list[Transaction]":  # noqa: F821 — avoid circular import at module level
    """
    Parse all transactions from a natural-language *message*.

    Returns a list of Transaction objects (possibly empty).
    _quick_filter avoids the LLM call for most non-transaction messages.
    """
    from app.state import Transaction, TransactionCategory
    from langchain_anthropic import ChatAnthropic
    from app.config import resolve_model_name, get_prompt
    from app.tools.cache import get_cached_llm_response, cache_llm_response

    if not _quick_filter(message):
        return []

    cached = get_cached_llm_response("transaction_extractor", message)
    if cached is not None and current_date is None:
        return [Transaction(**t) for t in cached.get("transactions", [])]

    try:
        model = resolve_model_name(os.getenv("SMARTFIN_MODEL", "claude-haiku-4-5-20251001"))
        llm = ChatAnthropic(model=model)
        prompt_messages = get_prompt("transaction_extractor").format_messages(message=message)
        result: _TxnExtract = llm.with_structured_output(_TxnExtract).invoke(prompt_messages)
    except Exception:
        return []

    try:
        txn_date = datetime.fromisoformat(current_date) if current_date else datetime.utcnow()
    except (ValueError, TypeError):
        txn_date = datetime.utcnow()

    extracted: list[Transaction] = []
    for item in result.transactions:
        try:
            category = TransactionCategory(item.category) if item.category else TransactionCategory.OTHER
        except ValueError:
            category = TransactionCategory.OTHER

        extracted.append(Transaction(
            id=f"chat-{uuid.uuid4().hex[:8]}",
            date=txn_date,
            amount=abs(item.amount),
            description=item.description or message[:100],
            merchant=item.merchant or "Unknown",
            category=category,
        ))

    if extracted:
        cache_llm_response(
            "transaction_extractor",
            message,
            {"transactions": [t.model_dump(mode="json") for t in extracted]},
        )

    return extracted
