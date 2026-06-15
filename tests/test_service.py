"""Tests for the Phase 6 service + API layer.

The chart heuristic and database discovery are pure and tested directly. The API
is exercised end-to-end with FastAPI's TestClient, mocking the LLM so no network
is needed — this proves the whole HTTP path (request -> agent -> execution ->
JSON response) wires up correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fastapi.testclient import TestClient

import make_sample_db  # noqa: E402
from text2sql import agent, analyst  # noqa: E402
from text2sql.llm import LLMResponse  # noqa: E402
from text2sql.service import suggest_chart  # noqa: E402


def test_suggest_chart_for_label_value():
    cols = ["city", "n"]
    rows = [("NY", 10), ("LA", 7), ("SF", 5)]
    chart = suggest_chart(cols, rows)
    assert chart and chart["type"] == "bar"
    assert chart["x"] == ["NY", "LA", "SF"]
    assert chart["y"] == [10, 7, 5]


def test_no_chart_for_single_aggregate():
    assert suggest_chart(["count"], [(42,)]) is None


def test_no_chart_without_numeric_column():
    assert suggest_chart(["a", "b"], [("x", "y"), ("p", "q")]) is None


def test_chart_skips_id_picks_measure():
    """Bug fix: with an id column present, bar height must be the measure, not id."""
    cols = ["id", "name", "sales"]
    rows = [(101, "A", 30), (102, "B", 90), (103, "C", 60)]
    chart = suggest_chart(cols, rows)
    assert chart["y_label"] == "sales"  # not "id"
    assert chart["x"] == ["B", "C", "A"]  # sorted by sales desc
    assert chart["y"] == [90, 60, 30]


def test_chart_sorts_descending_without_order_by():
    cols = ["city", "n"]
    rows = [("NY", 5), ("LA", 12), ("SF", 8)]
    chart = suggest_chart(cols, rows)  # no SQL -> default desc sort
    assert chart["x"] == ["LA", "SF", "NY"]
    assert chart["y"] == [12, 8, 5]


def test_chart_respects_explicit_order_by():
    """A time-series/ranking the agent already ordered must be left as-is."""
    cols = ["month", "revenue"]
    rows = [("Jan", 30), ("Feb", 50), ("Mar", 20)]
    chart = suggest_chart(cols, rows, sql="SELECT month, revenue FROM t ORDER BY month")
    assert chart["x"] == ["Jan", "Feb", "Mar"]  # original order preserved
    assert chart["y"] == [30, 50, 20]


def _client(monkeypatch):
    make_sample_db.build()  # ensure data/sample.sqlite exists

    def fake_generate(messages, *, temperature=None, json_mode=False, **kwargs):
        sql = "SELECT city, COUNT(*) AS n FROM customers GROUP BY city"
        return LLMResponse(
            content=json.dumps({"action": "sql", "reasoning": "group by city", "sql": sql}),
            prompt_tokens=120,
            completion_tokens=20,
            model="mock",
        )

    def fake_analyst(messages, *, temperature=None, json_mode=False, **kwargs):
        return LLMResponse(
            content=json.dumps(
                {"answer": "客户主要分布在几个城市。", "insights": ["最多的城市占比最高"]}
            ),
            prompt_tokens=80,
            completion_tokens=15,
            model="mock",
        )

    monkeypatch.setattr(agent, "complete", fake_generate)  # SQL generation pass
    monkeypatch.setattr(analyst, "complete", fake_analyst)  # answer/insight pass
    from text2sql.api import app

    return TestClient(app)


def test_api_databases_lists_sample(monkeypatch):
    client = _client(monkeypatch)
    dbs = client.get("/api/databases").json()
    assert any(d["id"] == "sample" for d in dbs)


def test_api_ask_end_to_end(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/api/ask", json={"question": "customers per city", "db": "sample", "schema": "lexical"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "GROUP BY city" in body["sql"]
    assert body["row_count"] >= 1
    assert body["chart"] is not None  # city + count -> bar chart suggested
    assert body["model"] == "mock"
    assert body["answer"] == "客户主要分布在几个城市。"  # P7: natural-language answer
    assert body["insights"]  # at least one insight
    assert body["tokens"]["total"] == 235  # 120+20 generate + 80+15 analyst


def test_api_ask_without_analysis(monkeypatch):
    client = _client(monkeypatch)
    r = client.post(
        "/api/ask", json={"question": "customers per city", "db": "sample", "analyze": False}
    )
    body = r.json()
    assert body["ok"] is True
    assert body["answer"] is None  # analysis disabled -> no narrated answer
    assert body["tokens"]["total"] == 140  # only the generation pass counted


def test_api_ask_returns_clarification(monkeypatch):
    """P8: an ambiguous question yields a clarify turn, not SQL/execution."""
    make_sample_db.build()

    def fake_clarify(messages, *, temperature=None, json_mode=False, **kwargs):
        return LLMResponse(
            content=json.dumps({"action": "clarify", "clarification": "你指的是哪一年的数据?"}),
            prompt_tokens=50,
            completion_tokens=10,
            model="mock",
        )

    monkeypatch.setattr(agent, "complete", fake_clarify)
    from text2sql.api import app

    client = TestClient(app)

    r = client.post("/api/ask", json={"question": "销量怎么样", "db": "sample"})
    body = r.json()
    assert body["action"] == "clarify"
    assert "哪一年" in body["clarification"]
    assert not body.get("sql")  # no SQL produced, nothing executed


def test_api_ask_accepts_history(monkeypatch):
    """A follow-up request carrying conversation history still resolves to SQL."""
    client = _client(monkeypatch)
    r = client.post(
        "/api/ask",
        json={
            "question": "只看北京",
            "db": "sample",
            "history": [
                {
                    "question": "每个城市有多少客户",
                    "sql": "SELECT city, COUNT(*) FROM customers GROUP BY city",
                }
            ],
        },
    )
    body = r.json()
    assert body["action"] == "sql"
    assert body["ok"] is True
