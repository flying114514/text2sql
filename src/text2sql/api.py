"""FastAPI app (Phase 6): serves the agent over HTTP + the single-page frontend.

    uv run python scripts/serve.py      # then open http://127.0.0.1:8000

Endpoints:
    GET  /                  -> the web UI (static single page)
    GET  /api/databases     -> databases the UI can query
    POST /api/ask           -> run the agent for one question
    GET  /api/traces/summary-> lightweight observability stats (Phase 5)
    GET  /api/gateway/status -> live gateway runtime: breaker state, strategy (P13)
    GET  /api/gateway/metrics-> per-provider/model + time-series trace aggregates (P13)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import cache, feedback, pins, service
from .gateway import gateway
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
    provider_id: str | None = None

    model_config = {"populate_by_name": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/chart.umd.min.js")
def chart_js() -> FileResponse:
    return FileResponse(WEB_DIR / "chart.umd.min.js", media_type="text/javascript")


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
        provider_id=req.provider_id,
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


def _load_events(day: str | None = None) -> list[dict]:
    """Read trace JSONL events (all days, or one YYYYMMDD). Shared by the summary
    chip and the gateway monitoring page."""
    pattern = f"llm-{day}.jsonl" if day else "llm-*.jsonl"
    events: list[dict] = []
    if not TRACES_DIR.exists():
        return events
    for f in sorted(TRACES_DIR.glob(pattern)):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def _group_stats(events: list[dict], key_fn) -> list[dict]:
    """Aggregate events into per-group rows (calls/ok_rate/tokens/cost/p50/p95).

    Mirrors scripts/trace_report._group_report, but returns JSON for the UI."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        groups[key_fn(e)].append(e)
    rows = []
    for name, evs in groups.items():
        calls = len(evs)
        ok = sum(1 for e in evs if e.get("ok"))
        lat = [e.get("latency_s", 0.0) for e in evs]
        rows.append(
            {
                "name": name,
                "calls": calls,
                "ok_rate": round(ok / calls, 4) if calls else None,
                "tokens": sum(e.get("total_tokens", 0) for e in evs),
                "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in evs), 6),
                "p50_s": round(_percentile(lat, 50), 4),
                "p95_s": round(_percentile(lat, 95), 4),
            }
        )
    rows.sort(key=lambda r: r["calls"], reverse=True)
    return rows


@app.get("/api/traces/summary")
def traces_summary() -> dict:
    """Aggregate the local trace logs into a small dashboard payload."""
    calls = ok = tokens = 0
    cost = 0.0
    for e in _load_events():
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


@app.get("/api/gateway/status")
def gateway_status() -> dict:
    """Live gateway runtime: routing strategy, each provider's circuit-breaker
    state, and per-role limits (P13). Read-only — never triggers a call."""
    return gateway.runtime_status()


@app.get("/api/gateway/metrics")
def gateway_metrics(day: str | None = None, limit: int = 50) -> dict:
    """Aggregate trace events for the monitoring page: per-provider and per-model
    tables, an hourly time series, and the most recent N calls."""
    events = _load_events(day)

    # Hourly buckets (by the HH of the ISO timestamp) — call volume, cost, latency.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        ts = str(e.get("ts", ""))
        bucket = f"{ts[11:13]}:00" if len(ts) >= 13 else "?"
        buckets[bucket].append(e)
    series = []
    for label in sorted(buckets):
        evs = buckets[label]
        lat = [e.get("latency_s", 0.0) for e in evs]
        series.append(
            {
                "bucket": label,
                "calls": len(evs),
                "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in evs), 6),
                "avg_latency_s": round(sum(lat) / len(lat), 4) if lat else 0.0,
            }
        )

    recent = [
        {
            "ts": e.get("ts"),
            "provider": e.get("provider") or "(legacy)",
            "model": e.get("model"),
            "ok": e.get("ok"),
            "total_tokens": e.get("total_tokens", 0),
            "cost_usd": e.get("cost_usd", 0.0),
            "latency_s": e.get("latency_s", 0.0),
            "error": e.get("error"),
        }
        for e in events[-max(0, limit) :][::-1]  # last N, newest first
    ]

    return {
        "by_provider": _group_stats(events, lambda e: e.get("provider") or "(legacy)"),
        "by_model": _group_stats(events, lambda e: e.get("model") or "?"),
        "series": series,
        "recent": recent,
        "total_calls": len(events),
    }
