"""Tests for Phase 5 reliability + observability: guard-in-db, query timeout,
trace logging, and model fallback. No real network calls."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import llm, tracing  # noqa: E402
from text2sql.db import execute_sql  # noqa: E402
from text2sql.llm import LLMResponse  # noqa: E402


# --- guard wired into execute_sql ------------------------------------------
def test_execute_blocks_write(tmp_path: Path):
    db = tmp_path / "x.sqlite"
    sqlite3.connect(db).executescript("CREATE TABLE t (a)").close()
    res = execute_sql(db, "DELETE FROM t")
    assert not res.ok and "blocked" in res.error


# --- query timeout ----------------------------------------------------------
def test_runaway_query_times_out(tmp_path: Path):
    db = tmp_path / "x.sqlite"
    sqlite3.connect(db).executescript("CREATE TABLE t (a)").close()
    # An infinite recursive CTE — only the watchdog can stop it.
    runaway = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT count(*) FROM r"
    t0 = time.perf_counter()
    res = execute_sql(db, runaway, timeout_s=1.0)
    elapsed = time.perf_counter() - t0
    assert not res.ok
    assert elapsed < 5.0  # interrupted well before any natural end
    assert "interrupt" in res.error.lower() or "OperationalError" in res.error


# --- local trace logging ----------------------------------------------------
def test_trace_records_event(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACES_DIR", tmp_path)
    tracing.record_llm_call(
        kind="completion",
        model="deepseek-chat",
        prompt_tokens=1000,
        completion_tokens=500,
        latency_s=1.23,
    )
    files = list(tmp_path.glob("llm-*.jsonl"))
    assert len(files) == 1
    event = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert event["model"] == "deepseek-chat"
    assert event["total_tokens"] == 1500
    assert event["cost_usd"] > 0  # 1000 in + 500 out, priced > 0


# --- model fallback ---------------------------------------------------------
def test_fallback_used_when_primary_fails(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_model", "primary")
    monkeypatch.setattr(llm.settings, "fallback_model", "backup")
    monkeypatch.setattr(llm, "get_client", lambda: "PRIMARY")
    monkeypatch.setattr(llm, "get_fallback_client", lambda: "BACKUP")
    monkeypatch.setattr(llm, "record_llm_call", lambda **k: None)  # no trace files

    seen = []

    def fake_call(client, model, messages, temperature, json_mode):
        seen.append(model)
        if model == "primary":
            raise RuntimeError("primary provider down")
        return LLMResponse(content="ok", prompt_tokens=1, completion_tokens=1, model=model)

    monkeypatch.setattr(llm, "_call", fake_call)

    resp = llm.complete([{"role": "user", "content": "hi"}])
    assert resp.model == "backup" and resp.content == "ok"
    assert seen == ["primary", "backup"]  # tried primary first, then fell back


def test_no_fallback_when_primary_ok(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_model", "primary")
    monkeypatch.setattr(llm.settings, "fallback_model", "")  # fallback disabled
    monkeypatch.setattr(llm, "get_client", lambda: "PRIMARY")
    monkeypatch.setattr(llm, "record_llm_call", lambda **k: None)

    seen = []

    def fake_call(client, model, messages, temperature, json_mode):
        seen.append(model)
        return LLMResponse(content="ok", prompt_tokens=1, completion_tokens=1, model=model)

    monkeypatch.setattr(llm, "_call", fake_call)

    resp = llm.complete([{"role": "user", "content": "hi"}])
    assert resp.model == "primary"
    assert seen == ["primary"]
