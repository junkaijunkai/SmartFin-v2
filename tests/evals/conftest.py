"""Shared fixtures for capability eval tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.state import Transaction, TransactionCategory
from tests.evals.provider import (
    OpenAICompatibleJudge,
    missing_eval_env,
    monkeypatch_eval_llm,
)
from tests.evals.reporting import write_summary


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "eval: LLM capability evaluation tests (exclude with -m 'not eval')"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = missing_eval_env()
    if not missing:
        return
    reason = "Capability eval provider not configured: " + ", ".join(missing)
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    write_summary()


@pytest.fixture(scope="session")
def judge() -> OpenAICompatibleJudge:
    return OpenAICompatibleJudge()


@pytest.fixture(autouse=True)
def _use_eval_provider(monkeypatch):
    monkeypatch_eval_llm(monkeypatch)


def _txn(
    txn_id: str,
    merchant: str,
    description: str,
    amount: float,
    category: TransactionCategory = TransactionCategory.OTHER,
    days_ago: int = 5,
) -> Transaction:
    dt = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    return Transaction(
        id=txn_id,
        date=dt,
        amount=amount,
        description=description,
        merchant=merchant,
        category=category,
    )
