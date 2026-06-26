from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.observability import init_trace, log_trace_event
from app.observability.events import API_REQUEST, ERROR_CATEGORISED, INTERNAL_ERROR
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from app.config import configure_logging, get_default_model_name, get_monitoring_settings
from app.orchestrator import app_graph
from app.state import Transaction, TransactionCategory

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_transactions.json"


class TransactionPayload(BaseModel):
    id: str
    date: str
    amount: float
    description: str
    merchant: str
    category: str = "other"
    location: str | None = None


class AnalyzeRequest(BaseModel):
    message: str
    thread_id: str | None = None
    monthly_income: float = 3200.0
    current_date: str | None = None
    use_sample_data: bool = True
    transactions: list[TransactionPayload] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    thread_id: str
    active_agent: str | None = None
    assistant_message: str | None = None
    pending_confirmation: dict | None = None
    alerts: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 2b – HTTP API models
# ---------------------------------------------------------------------------

class RunStreamRequest(BaseModel):
    message: str | None = None
    monthly_income: float = 3200.0
    current_date: str | None = None
    use_sample_data: bool = False
    transactions: list[TransactionPayload] = Field(default_factory=list)
    checkpoint_id: str | None = None
    resume: dict | None = None  # HITL resume: {"confirmed": bool, "message": str|None}


class PatchStateRequest(BaseModel):
    pending_confirmation: dict | None = None
    messages: list[dict] | None = None


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert Pydantic models / LangChain messages to plain JSON types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


app = FastAPI(title="SmartFin Backend", version="0.1.0")


# ---------------------------------------------------------------------------
# Observability middleware — trace_id for every HTTP request
# ---------------------------------------------------------------------------


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    init_trace()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - t0) * 1000
    log_trace_event(
        API_REQUEST,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    return response


def _safe_category(raw_value: str) -> TransactionCategory:
    try:
        return TransactionCategory(raw_value)
    except ValueError:
        return TransactionCategory.OTHER


def load_demo_transactions() -> list[Transaction]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        raw_transactions = json.load(handle)

    return [
        Transaction(
            id=item["id"],
            date=item["date"],
            amount=item["amount"],
            description=item["description"],
            merchant=item["merchant"],
            category=_safe_category(item.get("category", "other")),
            location=item.get("location"),
        )
        for item in raw_transactions
    ]


def parse_transactions(items: list[TransactionPayload]) -> list[Transaction]:
    return [
        Transaction(
            id=item.id,
            date=item.date,
            amount=item.amount,
            description=item.description,
            merchant=item.merchant,
            category=_safe_category(item.category),
            location=item.location,
        )
        for item in items
    ]


def extract_assistant_message(messages: list) -> str | None:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "ai":
            return getattr(message, "content", None)
    return None


def serialize_alerts(alerts: list) -> list[dict]:
    serialized: list[dict] = []
    for alert in alerts or []:
        if hasattr(alert, "model_dump"):
            serialized.append(alert.model_dump(mode="json"))
        else:
            serialized.append(dict(alert))
    return serialized


def summarize_state(state: dict) -> dict:
    health_summary = state.get("health_summary")
    health_rating = None
    if health_summary is not None:
        rating = getattr(health_summary, "rating", None)
        health_rating = getattr(rating, "value", rating)

    return {
        "categorised_transaction_count": len(state.get("categorised_transactions", [])),
        "spending_trend_count": len(state.get("spending_trends", [])),
        "goal_count": len(state.get("goals", [])),
        "anomaly_flag_count": len(state.get("anomaly_flags", [])),
        "health_rating": health_rating,
    }


def build_response(thread_id: str, state: dict) -> AnalyzeResponse:
    return AnalyzeResponse(
        thread_id=thread_id,
        active_agent=state.get("active_agent"),
        assistant_message=extract_assistant_message(state.get("messages", [])),
        pending_confirmation=state.get("pending_confirmation"),
        alerts=serialize_alerts(state.get("alerts", [])),
        summary=summarize_state(state),
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "smartfin-backend",
        "status": "ok",
        "default_model": get_default_model_name(),
    }


@app.get("/health")
def health() -> dict:
    health_status: dict = {
        "service": "smartfin-backend",
        "status": "ok",
        "default_model": get_default_model_name(),
        "monitoring": get_monitoring_settings(),
    }

    try:
        from app.orchestrator.checkpoints import _conn
        _conn.execute("SELECT 1")
        health_status["checkpoint_db"] = "ok"
    except Exception as exc:
        health_status["checkpoint_db"] = "unavailable"
        health_status["status"] = "degraded"
        logger.warning("Health check: checkpoint DB unavailable: %s", exc)

    return health_status


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    thread_id = request.thread_id or str(uuid4())

    # Sync handlers run in a thread pool where the async middleware's
    # contextvars are NOT visible, so initialise trace here.
    init_trace(thread_id=thread_id)

    transactions = parse_transactions(request.transactions)
    if not transactions and request.use_sample_data:
        transactions = load_demo_transactions()

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "monthly_income": request.monthly_income,
        "goals": [],
        "current_date": request.current_date or date.today().isoformat(),
    }
    if transactions:
        initial_state["transactions"] = transactions

    try:
        state = app_graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        logger.exception("SmartFin backend request failed.")
        log_trace_event(
            ERROR_CATEGORISED, error_category=INTERNAL_ERROR,
            endpoint="/analyze", message=str(exc)[:200],
        )
        raise HTTPException(status_code=500, detail="SmartFin backend request failed.") from exc

    return build_response(thread_id, state)


# ---------------------------------------------------------------------------
# Phase 2b endpoints — thread-level state + streaming
# ---------------------------------------------------------------------------

@app.get("/threads/{thread_id}/state")
def get_thread_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app_graph.get_state(config)

    if snapshot is None:
        return {"values": None, "next": [], "checkpoint_id": None}

    next_nodes = list(snapshot.next) if snapshot.next else []
    checkpoint_id = None
    if snapshot.config and "configurable" in snapshot.config:
        checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")

    values = _make_json_safe(snapshot.values) if snapshot.values else None

    return {"values": values, "next": next_nodes, "checkpoint_id": checkpoint_id}


@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: RunStreamRequest):
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
    }
    if request.checkpoint_id:
        config["configurable"]["checkpoint_id"] = request.checkpoint_id

    # Handle HITL resume atomically (no separate patch request).
    if request.resume is not None:
        snapshot = app_graph.get_state(config)
        current_pc = {}
        if snapshot and snapshot.values:
            current_pc = dict(snapshot.values.get("pending_confirmation") or {})

        update: dict[str, Any] = {
            "pending_confirmation": current_pc,
            "hitl_decision": {"confirmed": request.resume.get("confirmed", False)},
        }
        msg = request.resume.get("message")
        if msg:
            update["messages"] = [HumanMessage(content=msg)]
            update["active_agent"] = None
        app_graph.update_state(config, update)
        initial_state = None  # resume from checkpoint

    elif request.message:
        # Guard: if the graph is still paused from a previous run (Docker restart
        # interrupted a HITL or crashed mid-processing), reset routing so the new
        # message starts a fresh turn instead of being consumed by the stale flow.
        snap = app_graph.get_state(config)
        if snap and snap.next:
            app_graph.update_state(
                config,
                {
                    "active_agent": None,
                    "pending_confirmation": None,
                    "hitl_rollback": None,
                    "hitl_decision": None,
                    "pending_intent": None,
                },
                as_node="supervisor",
            )

        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=request.message)],
            "monthly_income": request.monthly_income,
            "current_date": request.current_date or date.today().isoformat(),
        }
        transactions = parse_transactions(request.transactions)
        if not transactions and request.use_sample_data:
            transactions = load_demo_transactions()
        if transactions:
            initial_state["transactions"] = transactions
    else:
        initial_state = None  # plain resume (clarification etc.)

    # Nodes whose internal LLM reasoning is not useful to surface (routing/utility).
    _REASONING_SKIP_NODES = frozenset({"memory_loader", "memory_saver", "supervisor"})

    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                blk.get("text", "") for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            )
        return ""

    def generate():
        init_trace(thread_id=thread_id)
        # Buffer for accumulating text chunks per node until a complete LLM turn.
        text_buffers: dict[str, str] = {}
        try:
            for mode, data in app_graph.stream(
                initial_state, config, stream_mode=["updates", "messages"]
            ):
                if mode == "messages":
                    msg_chunk, meta = data
                    node = meta.get("langgraph_node", "")
                    if node.startswith("__") or node in _REASONING_SKIP_NODES:
                        continue
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue

                    text = _extract_text(msg_chunk.content)
                    if text:
                        text_buffers[node] = text_buffers.get(node, "") + text

                    # Detect end of a complete LLM turn via stop_reason.
                    stop_reason = (
                        msg_chunk.response_metadata.get("stop_reason")
                        or msg_chunk.response_metadata.get("finish_reason")
                    )
                    if stop_reason and text_buffers.get(node):
                        step = text_buffers.pop(node).strip()
                        if step:
                            yield f"data: {json.dumps({'node': node, 'reasoning_step': step})}\n\n"

                elif mode == "updates":
                    for node, update in data.items():
                        if update is None or node.startswith("__"):
                            continue
                        safe = _make_json_safe(update)
                        yield f"data: {json.dumps({'node': node, 'updates': safe})}\n\n"

            # After the stream completes, check if the graph is paused for HITL.
            # Yield a __pause__ event so the UI can show the HITL card without
            # polling the state endpoint (eliminates the race condition).
            snapshot = app_graph.get_state(config)
            if snapshot and snapshot.next:
                pc = (snapshot.values or {}).get("pending_confirmation")
                yield f"data: {json.dumps({'node': '__pause__', 'updates': {'pending_confirmation': _make_json_safe(pc)}})}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Streaming graph execution failed.")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.patch("/threads/{thread_id}/state")
def patch_thread_state(thread_id: str, request: PatchStateRequest):
    config = {"configurable": {"thread_id": thread_id}}
    update: dict[str, Any] = {}

    if request.pending_confirmation is not None:
        snapshot = app_graph.get_state(config)

        # Preserve the original pending_confirmation fields so confirm_node
        # can dispatch correctly (action/agent/summary/details).
        current_pc = {}
        if snapshot and snapshot.values:
            current_pc = dict(snapshot.values.get("pending_confirmation") or {})

        # The confirmed flag must go into hitl_decision (a separate state field),
        # NOT into pending_confirmation.  confirm_node reads hitl_decision to
        # decide approve vs reject.  If we merged confirmed into pending_confirmation,
        # route_after_agent could also re-evaluate and skip confirm_node entirely.
        confirmed_flag = request.pending_confirmation.pop("confirmed", None)

        current_pc.update(request.pending_confirmation)
        update["pending_confirmation"] = current_pc

        if confirmed_flag is not None:
            update["hitl_decision"] = {"confirmed": confirmed_flag}

    if request.messages is not None:
        update["messages"] = [
            HumanMessage(content=m.get("content", ""))
            for m in request.messages
            if m.get("content")
        ]
        update["active_agent"] = None

    app_graph.update_state(config, update)
    return {"status": "ok"}


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    try:
        app_graph.checkpointer.delete_thread(thread_id)
    except Exception:
        pass
    return {"status": "ok"}
