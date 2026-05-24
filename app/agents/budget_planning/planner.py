from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def generate_budget_allocations(
    monthly_income: Optional[float],
    category_monthly_avg: Dict[str, float],
    category_trends: Dict[str, str],
    existing_budget: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    existing_budget = existing_budget or {}
    allocations: Dict[str, float] = {}

    for category, avg_spend in category_monthly_avg.items():
        avg_spend = max(float(avg_spend), 0.0)

        if category in existing_budget:
            allocations[category] = round(max(float(existing_budget[category]), 0.0), 2)
            continue

        trend = category_trends.get(category, "stable").lower()

        if trend == "fixed":
            budget = avg_spend
        elif trend == "rising":
            budget = avg_spend * 1.10
        elif trend == "volatile":
            budget = avg_spend * 0.95
        else:
            budget = avg_spend * 1.05

        allocations[category] = round(max(budget, 0.0), 2)

    for category, value in existing_budget.items():
        if category not in allocations:
            allocations[category] = round(max(float(value), 0.0), 2)

    total_budget = sum(allocations.values())

    if monthly_income is not None:
        monthly_income = max(float(monthly_income), 0.0)
        max_budget = monthly_income * 0.90

        if total_budget > max_budget and total_budget > 0:
            scale = max_budget / total_budget
            for category in allocations:
                allocations[category] = round(allocations[category] * scale, 2)

    return allocations


def calculate_monthly_spending(
    transactions: List[Dict[str, Any]] | List[Any]
) -> Dict[str, float]:
    """
    Calculate monthly spending by category from transaction list.

    Accepts both dict-based and object-based transactions.
    Only processes expenses (amount > 0); income entries are ignored.
    """
    spending: Dict[str, float] = defaultdict(float)

    for tx in transactions:
        try:
            # Handle both dict and object (Pydantic model) forms
            if isinstance(tx, dict):
                amount = float(tx.get("amount", 0.0))
                category = str(tx.get("category", "uncategorized")).strip().lower()
            else:
                # Pydantic Transaction object
                amount = float(tx.amount)
                category = str(tx.category.value).strip().lower() if hasattr(tx.category, 'value') else str(tx.category).strip().lower()
        except (TypeError, ValueError, AttributeError):
            continue

        if amount <= 0:
            continue

        if not category:
            category = "uncategorized"

        spending[category] += amount

    return {category: round(amount, 2) for category, amount in spending.items()}


def evaluate_budget_progress(
    budget_allocations: Dict[str, float],
    actual_spending: Dict[str, float],
    current_day: int,
    days_in_month: int
) -> Dict[str, Dict[str, Any]]:
    if days_in_month <= 0:
        raise ValueError("days_in_month must be greater than 0")

    progress: Dict[str, Dict[str, Any]] = {}
    current_day = max(1, min(current_day, days_in_month))
    expected_ratio = current_day / days_in_month

    all_categories = set(budget_allocations.keys()) | set(actual_spending.keys())

    for category in all_categories:
        budget = float(budget_allocations.get(category, 0.0))
        spent = float(actual_spending.get(category, 0.0))

        if budget <= 0:
            if spent > 0:
                usage_ratio = 1.0
                remaining = -spent
                status = "exceeded"
            else:
                usage_ratio = 0.0
                remaining = 0.0
                status = "on_track"
        else:
            usage_ratio = spent / budget
            remaining = budget - spent

            if usage_ratio >= 1.0:
                status = "exceeded"
            elif usage_ratio > expected_ratio + 0.10:
                status = "near_limit"
            else:
                status = "on_track"

        progress[category] = {
            "spent": round(spent, 2),
            "budget": round(budget, 2),
            "remaining": round(remaining, 2),
            "usage_ratio": round(usage_ratio, 3),
            "expected_ratio_by_today": round(expected_ratio, 3),
            "status": status,
        }

    return progress


def generate_budget_warnings(
    progress: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    for category, item in progress.items():
        usage_ratio = float(item.get("usage_ratio", 0.0))
        expected_ratio = float(item.get("expected_ratio_by_today", 0.0))
        status = item.get("status", "on_track")

        if status == "exceeded" or usage_ratio >= 1.0:
            warnings.append({
                "category": category,
                "severity": "high",
                "message": f"Spending in '{category}' has exceeded the monthly budget."
            })
        elif usage_ratio > expected_ratio + 0.25:
            warnings.append({
                "category": category,
                "severity": "high",
                "message": f"Spending in '{category}' is far ahead of schedule and is likely to exceed the monthly budget."
            })
        elif usage_ratio > expected_ratio + 0.15:
            warnings.append({
                "category": category,
                "severity": "medium",
                "message": f"Spending in '{category}' is ahead of the expected monthly pace."
            })
        elif usage_ratio >= 0.80:
            warnings.append({
                "category": category,
                "severity": "low",
                "message": f"Spending in '{category}' is approaching its monthly budget limit."
            })

    return warnings