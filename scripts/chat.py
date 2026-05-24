"""
Interactive CLI for testing SmartFin backend without the UI.

Usage:
    python scripts/chat.py

Commands during session:
    /approve    — approve the pending HITL action
    /reject     — reject the pending HITL action
    /quit       — exit
"""

import json
import uuid
from pathlib import Path
from datetime import date

from dotenv import load_dotenv
load_dotenv() # comment if testing service degradation without LLM calls


from langchain_core.messages import HumanMessage

from app.orchestrator import app_graph, get_pending_interrupt, resume_with_confirmation
from app.state import Transaction, TransactionCategory


FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_transactions.json"


def load_transactions() -> list[Transaction]:
    with open(FIXTURE_PATH) as f:
        raw = json.load(f)
    return [
        Transaction(
            id=t["id"],
            date=t["date"],
            amount=t["amount"],
            description=t["description"],
            merchant=t["merchant"],
            category=TransactionCategory(t["category"]),
        )
        for t in raw
    ]


def print_pending(pc: dict) -> None:
    print("\n" + "=" * 60)
    print(f"[HITL PAUSE]  {pc.get('summary', '')}")
    print("-" * 60)
    for detail in pc.get("details", []):
        print(f"  • {detail}")
    print(f"\nAction : {pc.get('action')}")
    print(f"Agent  : {pc.get('agent')}")
    print("=" * 60)
    print("Type a reply to clarify, /approve to confirm, or /reject to cancel.")


def print_state_summary(state: dict) -> None:
    print()
    if state.get("expense_analysis"):
        ea = state["expense_analysis"]
        total = sum(ea.get("category_monthly_avg", {}).values())
        print(f"[Expense Analysis] Last 30 days total: £{total:.2f}")

    if state.get("budget_summary"):
        print(f"[Budget Planning]  {state['budget_summary']}")

    if state.get("goals"):
        print(f"[Goal Planning]    {len(state['goals'])} goal(s) tracked")

    if state.get("anomaly_explanation"):
        print(f"[Anomaly Detection] {state['anomaly_explanation']}")
    elif state.get("anomaly_flags") is not None:
        flags = state.get("anomaly_flags") or []
        print(f"[Anomaly Detection] {len(flags)} anomaly flag(s) found.")

    if state.get("health_summary"):
        hs = state["health_summary"]
        rating = hs.rating.value if hasattr(hs.rating, "value") else hs.rating
        dti = f"{hs.debt_to_income_ratio:.0%}"
        reserves = f"{hs.liquid_reserve_months:.1f} months"
        print(f"[Health Assessment] Rating: {rating} | DTI: {dti} | Reserves: {reserves}")
        for obs in hs.observations:
            print(f"  • {obs}")

    alerts = state.get("alerts") or []
    if alerts:
        print(f"[Alerts] {len(alerts)} alert(s):")
        for a in alerts[-5:]:
            severity = a.severity.value if hasattr(a.severity, "value") else a.severity
            print(f"  [{severity.upper()}] {a.message}")

    msgs = state.get("messages", [])
    for msg in reversed(msgs):
        if hasattr(msg, "type") and msg.type == "ai":
            print(f"\nAssistant: {msg.content}")
            break


def main() -> None:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    transactions = load_transactions()

    print("SmartFin CLI — type your message, /approve, /reject, or /quit")
    print(f"Thread: {thread_id[:8]}...\n")

    first_turn = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break

        paused = get_pending_interrupt(app_graph, config)

        if user_input == "/approve":
            if not paused:
                print("[No pending HITL to approve]")
                continue
            state = resume_with_confirmation(app_graph, config, confirmed=True)
            print_state_summary(state)

        elif user_input == "/reject":
            if not paused:
                print("[No pending HITL to reject]")
                continue
            state = resume_with_confirmation(app_graph, config, confirmed=False)
            print_state_summary(state)

        elif paused:
            # Graph is paused — treat input as clarification message
            state = resume_with_confirmation(
                app_graph, config, confirmed=True, user_message=user_input
            )
            print_state_summary(state)

        else:
            # Normal turn — invoke graph fresh
            invoke_state: dict = {
                "messages": [HumanMessage(content=user_input)],
                "monthly_income": 3200.0,
                "goals": [],
                "current_date": date.today().isoformat(),
            }
            if first_turn:
                invoke_state["transactions"] = transactions
                first_turn = False

            state = app_graph.invoke(invoke_state, config)
            print_state_summary(state)

        # Check if still paused after this turn
        paused = get_pending_interrupt(app_graph, config)
        if paused:
            pc = paused.get("pending_confirmation", {})
            print_pending(pc)


if __name__ == "__main__":
    main()
