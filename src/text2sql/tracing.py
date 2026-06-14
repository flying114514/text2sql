"""Observability: trace every LLM call (Phase 5).

Each model call records one event — model, tokens, latency, estimated cost,
ok/error — to a date-partitioned JSONL file under data/traces/. This is the
local backend and is ALWAYS on: it needs no account and makes cost/latency
auditable offline (see scripts/trace_report.py for a summary).

If Langfuse credentials are configured, each event is *also* sent to Langfuse
(cloud UI) on a best-effort basis — failures there never affect the app. So the
same instrumentation works with or without the cloud platform.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

from .config import DATA_DIR
from .pricing import estimate_cost

TRACES_DIR = DATA_DIR / "traces"
_lock = threading.Lock()


def record_llm_call(
    *,
    kind: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    ok: bool = True,
    error: str | None = None,
) -> dict:
    """Record one LLM call to the local trace log (+ Langfuse if configured)."""
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "kind": kind,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_s": round(latency_s, 4),
        "cost_usd": round(estimate_cost(prompt_tokens, completion_tokens), 6),
        "ok": ok,
        "error": error,
    }
    _write_local(event)
    _maybe_langfuse(event)
    return event


def _write_local(event: dict) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(UTC).strftime("%Y%m%d")
    path = TRACES_DIR / f"llm-{day}.jsonl"
    line = json.dumps(event, ensure_ascii=False)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _maybe_langfuse(event: dict) -> None:
    """Best-effort forward to Langfuse; never raises into the caller."""
    from .config import settings

    if not settings.langfuse_ready():
        return
    try:  # pragma: no cover - exercised only when credentials are present
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        client.generation(
            name=event["kind"],
            model=event["model"],
            usage={
                "input": event["prompt_tokens"],
                "output": event["completion_tokens"],
                "total": event["total_tokens"],
            },
            metadata={
                "latency_s": event["latency_s"],
                "cost_usd": event["cost_usd"],
                "ok": event["ok"],
                "error": event["error"],
            },
        )
    except Exception:
        # Observability must never break the application path.
        pass
