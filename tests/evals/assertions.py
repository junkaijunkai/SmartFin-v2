"""Deterministic assertions for structured capability evals."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any


def _normalise_text(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).lower().replace("_", " ").split())


def assert_equal(case_id: str, field: str, actual: Any, expected: Any) -> None:
    assert actual == expected, (
        f"{case_id}: {field} mismatch; expected {expected!r}, got {actual!r}"
    )


def assert_float_equal(
    case_id: str,
    field: str,
    actual: float | int | None,
    expected: float | int | None,
    tolerance: float = 0.01,
) -> None:
    if actual is None or expected is None:
        assert actual == expected, (
            f"{case_id}: {field} mismatch; expected {expected!r}, got {actual!r}"
        )
        return
    assert abs(float(actual) - float(expected)) <= tolerance, (
        f"{case_id}: {field} mismatch; expected {expected!r}, got {actual!r}"
    )


def assert_date_equal(case_id: str, field: str, actual: Any, expected: str | None) -> None:
    actual_value = actual.isoformat() if isinstance(actual, date) else actual
    assert actual_value == expected, (
        f"{case_id}: {field} mismatch; expected {expected!r}, got {actual_value!r}"
    )


def assert_set_equal(
    case_id: str,
    field: str,
    actual: Iterable[Any],
    expected: Iterable[Any],
) -> None:
    actual_set = {_normalise_text(item) for item in actual}
    expected_set = {_normalise_text(item) for item in expected}
    assert actual_set == expected_set, (
        f"{case_id}: {field} mismatch; expected {sorted(expected_set)!r}, "
        f"got {sorted(actual_set)!r}"
    )


def assert_semantic_name(
    case_id: str,
    actual: str | None,
    expected: str | None,
    aliases: Iterable[str] = (),
) -> None:
    if expected is None:
        assert actual is None, (
            f"{case_id}: name mismatch; expected None, got {actual!r}"
        )
        return
    allowed = {_normalise_text(expected), *(_normalise_text(alias) for alias in aliases)}
    actual_norm = _normalise_text(actual)
    assert actual_norm in allowed, (
        f"{case_id}: name mismatch; expected one of {sorted(allowed)!r}, got {actual!r}"
    )
