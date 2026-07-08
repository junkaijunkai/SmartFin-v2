"""Golden dataset loading for capability evals."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GOLDENS_DIR = Path(__file__).parent / "goldens"
VALID_SUITES = {"smoke", "full"}


@dataclass(frozen=True)
class GoldenCase:
    id: str
    tier: str
    locale: str
    input: str
    expected: dict[str, Any]
    context: dict[str, Any]


def eval_suite() -> str:
    suite = os.getenv("SMARTFIN_EVAL_SUITE", "smoke").lower()
    if suite not in VALID_SUITES:
        raise ValueError(
            f"SMARTFIN_EVAL_SUITE must be one of {sorted(VALID_SUITES)}, got {suite!r}"
        )
    return suite


def load_goldens(name: str) -> list[GoldenCase]:
    path = GOLDENS_DIR / f"{name}.jsonl"
    suite = eval_suite()
    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            case = GoldenCase(
                id=payload["id"],
                tier=payload["tier"],
                locale=payload.get("locale", "en"),
                input=payload["input"],
                expected=payload["expected"],
                context=payload.get("context", {}),
            )
            if case.id in seen_ids:
                raise ValueError(f"Duplicate golden id {case.id!r} in {path}:{line_no}")
            seen_ids.add(case.id)
            if suite == "smoke" and case.tier != "smoke":
                continue
            cases.append(case)

    if not cases:
        raise ValueError(f"No {suite!r} golden cases loaded from {path}")
    return cases
