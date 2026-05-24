"""
Memory writer — persists agent outputs to Markdown files with YAML frontmatter.

Each memory file lives under ``.smartfin/memory/`` and follows a consistent
structure: YAML frontmatter (name, description, type) followed by the
content body in Markdown.

The index file ``MEMORY.md`` is automatically updated whenever a new file
is created, so the retriever always has a fresh manifest.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from app.state import (
    BudgetAllocation,
    FinancialGoal,
    Transaction,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEMORY_ROOT = _REPO_ROOT / ".smartfin" / "memory"
_INDEX_FILE = "MEMORY.md"


# ---------------------------------------------------------------------------
# Type → directory mapping
# ---------------------------------------------------------------------------

_MEMORY_DIRS: dict[str, str] = {
    "transaction": "transactions",
    "income": "incomes",
    "goal": "goals",
    "budget": "budgets",
}

# AppState field → memory type
_FIELD_TO_TYPE: dict[str, str] = {
    "transactions": "transaction",
    "monthly_income": "income",
    "goals": "goal",
    "budget_allocations": "budget",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_memory(thread_id: str, field: str, value: Any) -> None:
    """Persist a single AppState field to a memory file.

    Args:
        thread_id: Used only for logging; memory is global per user.
        field: AppState field name (e.g. ``"goals"``, ``"transactions"``).
        value: The field value (list of models, dict, etc.).
    """
    mem_type = _FIELD_TO_TYPE.get(field)
    if mem_type is None:
        logger.debug("[memory] No memory type for field '%s', skipping", field)
        return

    dir_name = _MEMORY_DIRS.get(mem_type)
    if dir_name is None:
        return

    dir_path = _MEMORY_ROOT / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path, description = _build_path_and_description(dir_path, mem_type, field, value)

    # Skip if value is empty
    if value is None or (isinstance(value, list) and not value):
        logger.info("[memory] value empty, skipping")
        return

    content = _serialize(field, value)
    frontmatter = {
        "name": str(file_path.relative_to(_MEMORY_ROOT).as_posix()),
        "description": description,
        "type": mem_type,
    }

    _write_md_file(file_path, frontmatter, content)
    _update_index(thread_id, frontmatter)


# ---------------------------------------------------------------------------
# Path & description builders
# ---------------------------------------------------------------------------


def _build_path_and_description(
    dir_path: Path, mem_type: str, field: str, value: Any,
) -> tuple[Path, str]:
    """Determine the file path and a one-line description for the value."""
    today = date.today().isoformat()

    if mem_type == "transaction":
        month_key = today[:7]
        txns: list[Transaction] = value if isinstance(value, list) else []
        total = sum(abs(t.amount) for t in txns)
        desc = f"{len(txns)} transactions, total spending ${total:,.2f}"
        return dir_path / f"{month_key}.md", desc

    if mem_type == "income":
        month_key = today[:7]
        income = value if isinstance(value, (int, float)) else 0
        desc = f"Monthly income: ${income:,.2f}"
        return dir_path / f"{month_key}.md", desc

    if mem_type == "goal":
        goals: list[FinancialGoal] = value if isinstance(value, list) else []
        if not goals:
            return dir_path / "untitled.md", "No goals"
        g = goals[-1]
        name = _safe_filename(g.name)
        desc = f"Goal: {g.name}, target ${g.target_amount:,.2f} by {g.target_date}, saved ${g.current_amount:,.2f}"
        return dir_path / f"{name}.md", desc

    if mem_type == "budget":
        month_key = today[:7]
        allocs: list[BudgetAllocation] = value if isinstance(value, list) else []
        total_allocated = sum(a.allocated_amount for a in allocs)
        desc = f"Budget plan for {month_key}: {len(allocs)} categories, ${total_allocated:,.2f} allocated"
        return dir_path / f"{month_key}-plan.md", desc

    return dir_path / "unknown.md", ""


def _safe_filename(name: str) -> str:
    """Convert a goal name like 'Emergency Fund' to 'emergency-fund'."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower()).strip("-") or "goal"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _serialize(field: str, value: Any) -> str:
    """Convert an AppState field value to Markdown body."""
    if field == "transactions":
        return _transactions_to_md(value)
    if field == "goals":
        return _goals_to_md(value)
    if field == "budget_allocations":
        return _budgets_to_md(value)
    if field == "monthly_income":
        return f"**Monthly Income:** ${float(value):,.2f}"
    return json.dumps(value, indent=2, default=str) if value else ""


def _transactions_to_md(txns: list) -> str:
    lines = ["| Date | Amount | Category | Merchant | Description |",
             "|------|--------|----------|----------|-------------|"]
    for t in txns:
        d = t.date.date() if hasattr(t.date, "date") else str(t.date)[:10]
        amt = f"${t.amount:,.2f}" if t.amount >= 0 else f"(${abs(t.amount):,.2f})"
        cat = t.category.value if hasattr(t.category, "value") else str(t.category)
        desc = (t.description or "")[:60]
        lines.append(f"| {d} | {amt} | {cat} | {t.merchant} | {desc} |")
    return "\n".join(lines)


def _goals_to_md(goals: list) -> str:
    parts = []
    for g in goals:
        name = g.name if hasattr(g, "name") else str(g)
        target = g.target_amount if hasattr(g, "target_amount") else 0
        current = g.current_amount if hasattr(g, "current_amount") else 0
        tgt_date = g.target_date if hasattr(g, "target_date") else "?"
        monthly = g.required_monthly_saving if hasattr(g, "required_monthly_saving") else 0
        on_track = g.on_track if hasattr(g, "on_track") else True
        parts.append(
            f"### {name}\n"
            f"- **Target:** ${target:,.2f} by {tgt_date}\n"
            f"- **Saved:** ${current:,.2f}\n"
            f"- **Monthly saving needed:** ${monthly:,.2f}\n"
            f"- **On track:** {'Yes' if on_track else 'No'}\n"
        )
    return "\n".join(parts)


def _budgets_to_md(allocs: list) -> str:
    lines = ["| Category | Allocated | Spent | Remaining | Utilisation |",
             "|----------|-----------|-------|-----------|-------------|"]
    for a in allocs:
        cat = a.category.value if hasattr(a.category, "value") else str(a.category)
        alloc = f"${a.allocated_amount:,.2f}" if hasattr(a, "allocated_amount") else "—"
        spent = f"${a.spent_amount:,.2f}" if hasattr(a, "spent_amount") else "—"
        rem = f"${a.remaining:,.2f}" if hasattr(a, "remaining") else "—"
        util = f"{a.utilisation_rate:.0%}" if hasattr(a, "utilisation_rate") else "—"
        lines.append(f"| {cat} | {alloc} | {spent} | {rem} | {util} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _write_md_file(file_path: Path, frontmatter: dict, content: str) -> None:
    """Write a Markdown file with YAML frontmatter."""
    yaml_lines = ["---"]
    for k, v in frontmatter.items():
        yaml_lines.append(f"{k}: {v}")
    yaml_lines.append("---")
    yaml_lines.append("")
    yaml_lines.append(content)
    file_path.write_text("\n".join(yaml_lines), encoding="utf-8")


def _update_index(thread_id: str, entry: dict) -> None:
    """Append or update the MEMORY.md index with a new entry.

    Each entry is a one-line YAML list item:
        ``- name: transactions/2026-05.md  description: "..."  type: transaction``
    """
    index_path = _MEMORY_ROOT / _INDEX_FILE
    if not index_path.exists():
        _MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        index_path.write_text("", encoding="utf-8")

    # Build the new index line
    desc = entry.get("description", "").replace('"', "'")
    line = f'- name: {entry["name"]}  description: "{desc}"  type: {entry["type"]}'

    # Read existing lines; replace if entry for same name already exists
    existing = index_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    replaced = False
    for old_line in existing:
        if old_line.lstrip().startswith("- name:") and entry["name"] in old_line:
            new_lines.append(line)
            replaced = True
        else:
            new_lines.append(old_line)
    if not replaced:
        new_lines.append(line)

    index_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.debug("[memory] Index updated: %s (%s)", entry["name"], entry["description"])
