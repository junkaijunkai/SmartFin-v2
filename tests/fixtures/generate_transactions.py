#!/usr/bin/env python3
"""
Generate sample_transactions.json for unit and integration tests.

Run from the project root:
    python tests/fixtures/generate_transactions.py
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> None:
    now = datetime.now(tz=timezone.utc)

    def tx(
        id_: str,
        days_ago: float,
        amount: float,
        description: str,
        merchant: str,
        category: str,
        location: str | None = "London",
    ) -> dict:
        return {
            "id": id_,
            "date": (now - timedelta(days=days_ago)).isoformat(),
            "amount": amount,
            "description": description,
            "merchant": merchant,
            "category": category,
            "location": location,
        }

    transactions = [
        # ── Current period (0–30 days ago) ──────────────────────────────────
        tx("t01",  2,   45.50, "Weekly grocery shop",       "Tesco",            "food"),
        tx("t02",  5,   12.80, "Coffee and sandwich",       "Pret A Manger",    "food"),
        tx("t03",  7,   38.00, "Restaurant dinner",         "Nando's",          "food"),
        tx("t04", 12,   25.00, "Grocery top-up",            "Sainsbury's",      "food"),
        tx("t05",  3,   22.40, "Monthly bus pass top-up",   "TfL",              "transport"),
        tx("t06", 10,   15.00, "Taxi to airport",           "Uber",             "transport"),
        tx("t07", 20,    8.50, "Single bus fare",           "TfL",              "transport"),
        tx("t08",  1, 1200.00, "Monthly rent",              "Landlord",         "housing"),
        tx("t09",  8,   65.00, "Cinema and dinner",         "Vue Cinema",       "entertainment"),
        tx("t10", 13,   29.99, "Streaming subscriptions",   "Spotify",          "entertainment"),
        tx("t11",  4,   89.99, "New trainers",              "Nike",             "shopping"),
        tx("t12", 18,   42.00, "Clothes",                   "H&M",              "shopping"),
        tx("t13",  6,   55.00, "Electricity bill",          "British Gas",      "utilities"),
        tx("t14", 22,   28.00, "Internet bill",             "BT",               "utilities"),
        # healthcare & education appear only in current period → deviation_pct = None
        tx("t15",  9,   35.00, "GP prescription",           "Boots Pharmacy",   "healthcare"),
        tx("t16", 25,  150.00, "Online course",             "Udemy",            "education"),
        # income (negative amount) — must be excluded from spending trends
        tx("t17", 15, -3200.00, "Monthly salary",           "Employer Ltd",     "income"),

        # ── Previous period (31–60 days ago) ────────────────────────────────
        tx("t18", 35,   52.30, "Weekly grocery shop",       "Tesco",            "food"),
        tx("t19", 38,   18.50, "Lunch",                     "Pret A Manger",    "food"),
        tx("t20", 50,   21.00, "Grocery top-up",            "Waitrose",         "food"),
        tx("t21", 42,   22.40, "Monthly bus pass top-up",   "TfL",              "transport"),
        tx("t22", 45,   12.00, "Taxi home",                 "Uber",             "transport"),
        tx("t23", 31, 1200.00, "Monthly rent",              "Landlord",         "housing"),
        tx("t24", 40,   40.00, "Theatre tickets",           "National Theatre", "entertainment"),
        tx("t25", 50,   29.99, "Streaming subscriptions",   "Spotify",          "entertainment"),
        tx("t26", 55,  110.00, "Electricity bill",          "British Gas",      "utilities"),
        tx("t27", 48,   60.00, "New shoes",                 "Clarks",           "shopping"),
        # income in previous period — also excluded
        tx("t28", 44, -3200.00, "Monthly salary",           "Employer Ltd",     "income"),
    ]

    out_path = Path(__file__).parent / "sample_transactions.json"
    out_path.write_text(json.dumps(transactions, indent=2, ensure_ascii=False))
    print(f"[OK] Written {len(transactions)} transactions -> {out_path}")


if __name__ == "__main__":
    main()
