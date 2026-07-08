"""Capability eval summary reporting for local runs and CI artifacts."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tests.evals.loaders import eval_suite
from tests.evals.provider import eval_judge_model_name, eval_model_name


REPORT_DIR = Path(os.getenv("SMARTFIN_EVAL_REPORT_DIR", "reports/evals"))
_RESULTS: list["EvalResult"] = []


@dataclass
class EvalResult:
    case_id: str
    capability: str
    passed: bool
    reason: str = ""
    score: float | None = None


def record_result(
    case_id: str,
    capability: str,
    passed: bool,
    reason: str = "",
    score: float | None = None,
) -> None:
    _RESULTS.append(EvalResult(case_id, capability, passed, reason, score))


def write_summary() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    by_capability: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "total": 0})
    failures: list[dict[str, Any]] = []

    for result in _RESULTS:
        bucket = by_capability[result.capability]
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
            failures.append(asdict(result))

    payload = {
        "suite": eval_suite(),
        "model": eval_model_name(),
        "judge_model": eval_judge_model_name(),
        "total": len(_RESULTS),
        "passed": sum(1 for result in _RESULTS if result.passed),
        "failed": sum(1 for result in _RESULTS if not result.passed),
        "capabilities": dict(sorted(by_capability.items())),
        "failures": failures,
    }

    (REPORT_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "summary.md").write_text(_to_markdown(payload), encoding="utf-8")


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Capability Eval Summary",
        "",
        f"- Suite: `{payload['suite']}`",
        f"- Model: `{payload['model']}`",
        f"- Judge model: `{payload['judge_model']}`",
        f"- Passed: {payload['passed']}/{payload['total']}",
        "",
        "## Capabilities",
        "",
        "| Capability | Passed | Failed | Total |",
        "|---|---:|---:|---:|",
    ]
    for capability, stats in payload["capabilities"].items():
        lines.append(
            f"| {capability} | {stats['passed']} | {stats['failed']} | {stats['total']} |"
        )

    lines.extend(["", "## Failures", ""])
    if not payload["failures"]:
        lines.append("No failures.")
    else:
        for failure in payload["failures"]:
            reason = failure.get("reason") or "No reason captured."
            lines.append(f"- `{failure['case_id']}` ({failure['capability']}): {reason}")
    lines.append("")
    return "\n".join(lines)
