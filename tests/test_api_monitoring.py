"""P13 网关监控页面：API 端点 + 聚合逻辑。纯确定性,无网络。"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from text2sql.api import app, _group_stats, _load_events, _percentile  # noqa: E402

client = TestClient(app)


# ── helpers ──────────────────────────────────────────────────────────────


def _fake_events(**kw):
    """Build a list of trace event dicts. Each entry in **kw is a key→value list."""
    keys = list(kw.keys())
    events = []
    for i in range(len(next(iter(kw.values())))):
        events.append({k: kw[k][i] for k in keys})
    return events


# ── percentile ───────────────────────────────────────────────────────────


def test_percentile_median():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_percentile_p95():
    """20 values: p95 at index round(0.95 * 19) = 18 → sorted[18] = 19."""
    v = list(range(1, 21))
    assert _percentile(v, 95) == 19


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


# ── _group_stats ──────────────────────────────────────────────────────────


def test_group_stats_basic():
    events = _fake_events(
        provider=["a", "a", "b"],
        model=["m"] * 3,
        ok=[True, False, True],
        total_tokens=[10, 20, 30],
        cost_usd=[0.01, 0.02, 0.03],
        latency_s=[1.0, 2.0, 3.0],
        ts=["2026-06-15T10:00:00Z"] * 3,
    )
    rows = _group_stats(events, lambda e: e["provider"])
    rows.sort(key=lambda r: r["name"])  # default sort is by calls desc
    assert len(rows) == 2

    a = next(r for r in rows if r["name"] == "a")
    assert a["calls"] == 2
    assert a["ok_rate"] == 0.5
    assert a["tokens"] == 30
    assert a["cost_usd"] == 0.03
    assert a["p50_s"] == 1.0  # nearest-rank p50 on [1.0,2.0] → index round(0.5*1)=0 → s[0]=1.0

    b = next(r for r in rows if r["name"] == "b")
    assert b["calls"] == 1
    assert b["ok_rate"] == 1.0


# ── /api/gateway/status ──────────────────────────────────────────────────


def test_gateway_status_degraded():
    """无 gateway.yaml → enabled=False。"""
    resp = client.get("/api/gateway/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["providers"] == []
    assert data["limits"] == {}


# ── /api/gateway/metrics ─────────────────────────────────────────────────


@pytest.fixture
def fake_trace_dir(tmp_path, monkeypatch):
    """Create a temporary trace directory with controllable events."""
    (tmp_path / "traces").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("text2sql.api.TRACES_DIR", tmp_path / "traces")
    return tmp_path / "traces"


def test_metrics_empty(fake_trace_dir):
    """No trace files → empty payload, no crash."""
    resp = client.get("/api/gateway/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_calls"] == 0
    assert data["by_provider"] == []
    assert data["series"] == []
    assert data["recent"] == []


def test_metrics_aggregation(fake_trace_dir):
    """Events across two providers → correct by_provider / by_model rows."""
    ts = datetime(2026, 6, 15, 10, 30, 0, tzinfo=UTC)
    events = [
        {"ts": ts.isoformat(), "provider": "deepseek", "model": "deepseek-chat", "ok": True, "total_tokens": 100, "cost_usd": 0.0001, "latency_s": 2.0},
        {"ts": ts.isoformat(), "provider": "deepseek", "model": "deepseek-chat", "ok": True, "total_tokens": 200, "cost_usd": 0.0002, "latency_s": 1.5},
        {"ts": ts.isoformat(), "provider": "qwen", "model": "qwen-plus", "ok": False, "total_tokens": 50, "cost_usd": 0.0, "latency_s": 5.0, "error": "APIConnectionError: timeout"},
    ]
    (fake_trace_dir / "llm-20260615.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    resp = client.get("/api/gateway/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_calls"] == 3

    bp = {r["name"]: r for r in data["by_provider"]}
    assert bp["deepseek"]["calls"] == 2
    assert bp["deepseek"]["ok_rate"] == 1.0
    assert bp["deepseek"]["tokens"] == 300
    assert bp["qwen"]["calls"] == 1
    assert bp["qwen"]["ok_rate"] == 0.0

    bm = {r["name"]: r for r in data["by_model"]}
    assert bm["deepseek-chat"]["calls"] == 2
    assert bm["qwen-plus"]["calls"] == 1


def test_metrics_recent_limit(fake_trace_dir):
    """recent returns last N events, newest first, capped at limit."""
    events = []
    for i in range(10):
        events.append({"ts": f"2026-06-15T10:{i:02d}:00Z", "provider": "p", "model": "m", "ok": True, "total_tokens": i, "cost_usd": 0.0, "latency_s": 1.0})
    (fake_trace_dir / "llm-20260615.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    resp = client.get("/api/gateway/metrics?limit=4")
    data = resp.json()
    assert len(data["recent"]) == 4
    # newest first → highest token (9) first
    assert data["recent"][0]["total_tokens"] == 9
    assert data["recent"][1]["total_tokens"] == 8


def test_metrics_series_bucketing(fake_trace_dir):
    """Events in different hours → series with correct buckets."""
    events = [
        {"ts": "2026-06-15T09:15:00Z", "provider": "a", "model": "m", "ok": True, "total_tokens": 10, "cost_usd": 0.01, "latency_s": 1.0},
        {"ts": "2026-06-15T09:45:00Z", "provider": "a", "model": "m", "ok": True, "total_tokens": 20, "cost_usd": 0.02, "latency_s": 2.0},
        {"ts": "2026-06-15T11:00:00Z", "provider": "b", "model": "n", "ok": False, "total_tokens": 30, "cost_usd": 0.03, "latency_s": 3.0},
    ]
    (fake_trace_dir / "llm-20260615.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    resp = client.get("/api/gateway/metrics")
    data = resp.json()
    buckets = {s["bucket"]: s for s in data["series"]}
    assert "09:00" in buckets
    assert buckets["09:00"]["calls"] == 2
    assert buckets["09:00"]["cost_usd"] == 0.03
    assert buckets["09:00"]["avg_latency_s"] == 1.5
    assert "11:00" in buckets
    assert buckets["11:00"]["calls"] == 1


def test_traces_summary_still_works():
    """Existing /api/traces/summary unchanged."""
    resp = client.get("/api/traces/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "calls" in data
    assert "ok_rate" in data
    assert "total_tokens" in data
    assert "total_cost_usd" in data
