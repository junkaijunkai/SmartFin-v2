from __future__ import annotations

import json
import logging

from app.config import (
    JsonLogFormatter,
    get_default_model_name,
    get_monitoring_settings,
    is_model_approved,
    load_model_registry,
    resolve_model_name,
)


def test_load_model_registry_contains_default_alias():
    registry = load_model_registry()
    assert registry["default_alias"] in registry["approved_models"]


def test_default_model_is_approved():
    assert is_model_approved(get_default_model_name()) is True


def test_resolve_model_name_supports_alias():
    assert resolve_model_name("default") == get_default_model_name()


def test_resolve_model_name_allows_explicit_model_when_not_strict():
    assert resolve_model_name("claude-opus-4-7", strict=False) == "claude-opus-4-7"


def test_resolve_model_name_falls_back_in_strict_mode():
    assert resolve_model_name("unapproved-model", strict=True) == get_default_model_name()


def test_monitoring_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "smartfin-ci")
    settings = get_monitoring_settings()
    assert settings["langsmith_tracing"] is True
    assert settings["langsmith_project"] == "smartfin-ci"


def test_json_log_formatter_outputs_json():
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="smartfin.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.agent = "expense_analysis"
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hello"
    assert payload["agent"] == "expense_analysis"
