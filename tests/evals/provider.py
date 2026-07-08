"""OpenAI-compatible provider helpers for capability evals."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM


DEFAULT_EVAL_MODEL = "deepseek-v4-pro"
SUPPORTED_PROVIDER = "openai-compatible"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def eval_provider_name() -> str:
    return _env("SMARTFIN_EVAL_PROVIDER", SUPPORTED_PROVIDER) or SUPPORTED_PROVIDER


def eval_model_name() -> str:
    return _env("SMARTFIN_EVAL_MODEL", DEFAULT_EVAL_MODEL) or DEFAULT_EVAL_MODEL


def eval_judge_model_name() -> str:
    return _env("SMARTFIN_EVAL_JUDGE_MODEL", eval_model_name()) or eval_model_name()


def missing_eval_env() -> list[str]:
    missing = []
    if eval_provider_name() != SUPPORTED_PROVIDER:
        missing.append("SMARTFIN_EVAL_PROVIDER=openai-compatible")
    for key in ("SMARTFIN_EVAL_BASE_URL", "SMARTFIN_EVAL_API_KEY"):
        if not _env(key):
            missing.append(key)
    return missing


def make_eval_llm(alias: str | None = None, **kwargs) -> Any:
    """Return the model under test, routed directly to the eval provider."""
    from langchain_openai import ChatOpenAI

    missing = missing_eval_env()
    if missing:
        raise RuntimeError(
            "Capability eval provider is not configured. Missing: "
            + ", ".join(missing)
        )

    params = {
        "model": eval_model_name(),
        "base_url": _env("SMARTFIN_EVAL_BASE_URL"),
        "api_key": _env("SMARTFIN_EVAL_API_KEY"),
        "temperature": kwargs.pop("temperature", 0),
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the configured OpenAI-compatible provider."""

    def load_model(self) -> Any:
        from langchain_openai import ChatOpenAI

        missing = missing_eval_env()
        if missing:
            raise RuntimeError(
                "Capability eval judge is not configured. Missing: "
                + ", ".join(missing)
            )
        return ChatOpenAI(
            model=eval_judge_model_name(),
            base_url=_env("SMARTFIN_EVAL_BASE_URL"),
            api_key=_env("SMARTFIN_EVAL_API_KEY"),
            temperature=0,
        )

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content

    def get_model_name(self) -> str:
        return eval_judge_model_name()


def monkeypatch_eval_llm(monkeypatch) -> Callable[..., Any]:
    """Patch all currently evaluated modules to use the eval provider."""
    from app.agents.anomaly_detection import extractor as anomaly_extractor
    from app.agents.budget_planning import extractor as budget_extractor
    from app.agents.expense_analysis import categoriser as expense_categoriser
    from app.agents.goal_planning import extractor as goal_extractor
    from app.agents.health_assessment import assessor as health_assessor
    from app.orchestrator import intent_classifier
    import app.config as app_config

    for module in (
        app_config,
        anomaly_extractor,
        budget_extractor,
        expense_categoriser,
        goal_extractor,
        health_assessor,
        intent_classifier,
    ):
        monkeypatch.setattr(module, "get_llm", make_eval_llm, raising=False)

    for module in (budget_extractor, goal_extractor, intent_classifier):
        monkeypatch.setattr(
            module,
            "get_cached_llm_response",
            lambda *args, **kwargs: None,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "cache_llm_response",
            lambda *args, **kwargs: None,
            raising=False,
        )

    return make_eval_llm
