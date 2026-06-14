"""Tests for the Phase 4 self-correction loop — no real LLM/network.

We fake the LLM so the first response is *broken* SQL (wrong column) and the
second is correct, then assert the agent loop observes the DB error, repairs,
and recovers. This is the deterministic proof that the loop works, independent
of any live model (which on Spider rarely errors at all).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql import agent  # noqa: E402
from text2sql.llm import LLMResponse  # noqa: E402


@pytest.fixture()
def customers_db(tmp_path: Path) -> Path:
    db = tmp_path / "c.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE customers (id INTEGER, name TEXT, city TEXT);"
        "INSERT INTO customers VALUES (1, 'A', 'NY'), (2, 'B', 'NY'), (3, 'C', 'LA');"
    )
    conn.commit()
    conn.close()
    return db


class _FakeLLM:
    """Returns queued JSON responses, one per call; records how many calls."""

    def __init__(self, sqls: list[str]) -> None:
        self._responses = [json.dumps({"reasoning": "r", "sql": s}) for s in sqls]
        self.calls = 0

    def __call__(self, messages, *, temperature=None, json_mode=False):
        content = self._responses[self.calls]
        self.calls += 1
        return LLMResponse(content=content, prompt_tokens=10, completion_tokens=5)


def test_recovers_from_bad_column(customers_db: Path, monkeypatch):
    fake = _FakeLLM(
        [
            "SELECT town FROM customers",  # wrong column -> sqlite error
            "SELECT city FROM customers",  # corrected
        ]
    )
    monkeypatch.setattr(agent, "complete", fake)

    gen, resp, execu, trace = agent.generate_with_correction(
        "list cities", customers_db, max_retries=2
    )

    assert trace.attempts == 2
    assert trace.recovered is True
    assert trace.final_ok is True
    assert execu.ok and execu.row_count == 3
    assert resp.total_tokens == 30  # tokens summed across both attempts
    assert len(trace.errors) == 1  # one failed attempt recorded


def test_no_retry_when_first_attempt_succeeds(customers_db: Path, monkeypatch):
    fake = _FakeLLM(["SELECT city FROM customers"])
    monkeypatch.setattr(agent, "complete", fake)

    _gen, _resp, execu, trace = agent.generate_with_correction(
        "list cities", customers_db, max_retries=2
    )

    assert fake.calls == 1  # did not waste extra calls
    assert trace.attempts == 1
    assert trace.recovered is False
    assert execu.ok


def test_gives_up_after_max_retries(customers_db: Path, monkeypatch):
    fake = _FakeLLM(
        [
            "SELECT town FROM customers",  # all three reference a bad column
            "SELECT village FROM customers",
            "SELECT hamlet FROM customers",
        ]
    )
    monkeypatch.setattr(agent, "complete", fake)

    _gen, _resp, execu, trace = agent.generate_with_correction(
        "list cities", customers_db, max_retries=2
    )

    assert fake.calls == 3  # 1 initial + 2 retries, then stop
    assert trace.attempts == 3
    assert trace.final_ok is False
    assert execu is not None and not execu.ok
