from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.guardrails.input_filter import filter_user_input
from app.orchestrator.intent_classifier import classify_intent


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    filter_result = filter_user_input(prompt)

    if not filter_result["allowed"]:
        result = {
            "blocked": True,
            "event_type": "blocked_input",
            "filter_result": filter_result,
            "agent": None,
        }
        return {"output": json.dumps(result)}

    safe_message = filter_result["sanitized_message"]
    agent = classify_intent(safe_message)

    result = {
        "blocked": False,
        "event_type": None,
        "filter_result": filter_result,
        "agent": agent,
    }
    return {"output": json.dumps(result)}