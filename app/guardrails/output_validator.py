from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from app.state import BudgetAllocation


_OUTPUT_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}", re.IGNORECASE),
    "generic_api_key": re.compile(r"api[_ -]?key\s*[:=]\s*[A-Za-z0-9_-]{12,}", re.IGNORECASE),
    "bearer_token": re.compile(r"bearer\s+[A-Za-z0-9._-]{12,}", re.IGNORECASE),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}


@dataclass(frozen=True)
class OutputValidationResult:
    allowed: bool
    sanitized_text: str
    matched_rules: tuple[str, ...]


def validate_output(text: str) -> OutputValidationResult:
    """
    Generic text-output guardrail.

    Detects sensitive tokens / secrets / card-like numbers in text output
    and returns a redacted version if needed.
    """
    sanitized = text or ""
    matched_rules: list[str] = []

    for rule_name, pattern in _OUTPUT_PATTERNS.items():
        if pattern.search(sanitized):
            matched_rules.append(rule_name)
            sanitized = pattern.sub("[REDACTED]", sanitized)

    return OutputValidationResult(
        allowed=not matched_rules,
        sanitized_text=sanitized,
        matched_rules=tuple(matched_rules),
    )


def validate_budget_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate budget-planning agent output before it is trusted by the graph/state.

    Expected keys:
      - budget_allocations: list[BudgetAllocation]
      - budget_progress: dict
      - budget_warnings: list
      - budget_summary: str
      - budget_request: dict
    """
    errors: List[str] = []

    budget_allocations = result.get("budget_allocations")
    budget_progress = result.get("budget_progress")
    budget_warnings = result.get("budget_warnings")
    budget_summary = result.get("budget_summary")
    budget_request = result.get("budget_request")

    if budget_allocations is None:
        errors.append("missing_budget_allocations")
    elif not isinstance(budget_allocations, list):
        errors.append("budget_allocations_must_be_list")
    else:
        for i, item in enumerate(budget_allocations):
            if not isinstance(item, BudgetAllocation):
                errors.append(f"budget_allocations[{i}]_must_be_BudgetAllocation")
            else:
                if item.allocated_amount < 0:
                    errors.append(f"budget_allocations[{i}]_negative_allocated_amount")
                if item.spent_amount < 0:
                    errors.append(f"budget_allocations[{i}]_negative_spent_amount")

    if budget_progress is None:
        errors.append("missing_budget_progress")
    elif not isinstance(budget_progress, dict):
        errors.append("budget_progress_must_be_dict")

    if budget_warnings is None:
        errors.append("missing_budget_warnings")
    elif not isinstance(budget_warnings, list):
        errors.append("budget_warnings_must_be_list")
    else:
        for i, warning in enumerate(budget_warnings):
            if not isinstance(warning, dict):
                errors.append(f"budget_warnings[{i}]_must_be_dict")
                continue
            for key in ["category", "severity", "message"]:
                if key not in warning:
                    errors.append(f"budget_warnings[{i}]_missing_{key}")

    if budget_summary is None:
        errors.append("missing_budget_summary")
    elif not isinstance(budget_summary, str):
        errors.append("budget_summary_must_be_str")
    else:
        summary_check = validate_output(budget_summary)
        if not summary_check.allowed:
            errors.append("budget_summary_contains_sensitive_output")

    if budget_request is None:
        errors.append("missing_budget_request")
    elif not isinstance(budget_request, dict):
        errors.append("budget_request_must_be_dict")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "sanitized_output": result if len(errors) == 0 else None,
    }