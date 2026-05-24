"""Shared fixtures for deepeval LLM evaluation tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from deepeval.models.base_model import DeepEvalBaseLLM

from app.state import Transaction, TransactionCategory

load_dotenv()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "eval: deepeval LLM evaluation tests (exclude with -m 'not eval')"
    )


class ClaudeJudge(DeepEvalBaseLLM):
    """Thin wrapper so deepeval uses Claude as its judge model instead of OpenAI."""

    def load_model(self) -> ChatAnthropic:
        return ChatAnthropic(model="claude-haiku-4-5")

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content

    def get_model_name(self) -> str:
        return "claude-haiku-4-5"


@pytest.fixture(scope="session")
def judge() -> ClaudeJudge:
    return ClaudeJudge()


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
