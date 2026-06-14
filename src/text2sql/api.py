"""FastAPI app (Phase 6): serves the agent over HTTP + the single-page frontend.

    uv run python scripts/serve.py      # then open http://127.0.0.1:8000

Endpoints:
    GET  /                  -> the web UI (static single page)
    GET  /api/databases     -> databases the UI can query
    POST /api/ask           -> run the agent for one question
    GET  /api/traces/summary-> lightweight observability stats (Phase 5)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import cache, feedback, pins, service
from .governance import list_roles

WEB_DIR = Path(__file__).resolve().parent / "web"
TRACES_DIR = Path(__file__).resolve().parents[2] / "data" / "traces"

app = FastAPI(title="Text2SQL Agent", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    db: str = "sample"
    schema_mode: str = Field(default="lexical", alias="schema")
    fewshot: int = 0
    correct: int = 0
    analyze: bool = True
    history: list[dict] = Field(default_factory=list)
    role: str = ""
    use_cache: bool = True

    model_config = {"populate_by_name": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/databases")
def databases() -> list[dict]:
    return service.list_databases()


@app.get("/api/roles")
def roles() -> list[dict]:
    """Governance roles the UI offers in its identity switcher (P10)."""
    return list_roles()


@app.post("/api/ask")
def ask(
    req: AskRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> JSONResponse:
    result = service.answer_query(
        req.question,
        req.db,
        schema=req.schema_mode,
        fewshot=req.fewshot,
        correct=req.correct,
        analyze=req.analyze,
        history=req.history,
        role=req.role or None,
        api_key=x_api_key,
        use_cache=req.use_cache,
    )
    return JSONResponse(result)


@app.get("/api/cache/stats")
def cache_stats() -> dict:
    """How much the answer cache is saving (P11)."""
    return cache.stats()


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1)
    db: str = "sample"
    sql: str = ""
    rating: str  # "up" | "down"
    role: str = ""


@app.post("/api/feedback")
def feedback_ep(req: FeedbackRequest) -> dict:
    """Record a 👍/👎 (P11b). 👍 adopts the (question, SQL) as a verified same-db
    few-shot example; 👎 logs the miss and invalidates that question's cache."""
    return feedback.record(req.question, req.db, req.sql, req.rating, role=req.role or None)


@app.get("/api/feedback/stats")
def feedback_stats() -> dict:
    """What the feedback flywheel has accumulated (P11b)."""
    return feedback.stats()


class PinRequest(BaseModel):
    question: str = Field(min_length=1)
    db: str = "sample"
    role: str = ""
    label: str = ""
    answer: str = ""


@app.get("/api/pins")
def pins_list() -> list[dict]:
    """The dashboard: every pinned question (P11c)."""
    return pins.list_pins()


@app.post("/api/pins")
def pins_add(req: PinRequest) -> dict:
    """Pin a question into the dashboard (or refresh its preview). The pin is
    bound to the db + role so a card always re-runs as the identity that made it."""
    return pins.add(
        req.question,
        req.db,
        role=req.role or None,
        label=req.label or None,
        answer=req.answer if req.answer != "" else None,
    )


@app.delete("/api/pins/{pin_id}")
def pins_remove(pin_id: str) -> dict:
    return pins.remove(pin_id)


@app.get("/api/traces/summary")
def traces_summary() -> dict:
    """Aggregate the local trace logs into a small dashboard payload."""
    calls = ok = tokens = 0
    cost = 0.0
    if TRACES_DIR.exists():
        for f in sorted(TRACES_DIR.glob("llm-*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                calls += 1
                ok += 1 if e.get("ok") else 0
                tokens += e.get("total_tokens", 0)
                cost += e.get("cost_usd", 0.0)
    return {
        "calls": calls,
        "ok_rate": round(ok / calls, 4) if calls else None,
        "total_tokens": tokens,
        "total_cost_usd": round(cost, 4),
    }
