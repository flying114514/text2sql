"""Tests for the SQLite execution layer (no LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_sample_db  # noqa: E402
from text2sql.db import execute_sql


def _db() -> Path:
    return make_sample_db.build()


def test_valid_query_returns_rows():
    res = execute_sql(_db(), "SELECT COUNT(*) FROM customers")
    assert res.ok
    assert res.rows[0][0] == 4


def test_broken_query_fails_gracefully():
    res = execute_sql(_db(), "SELECT does_not_exist FROM customers")
    assert not res.ok
    assert res.error is not None


def test_readonly_blocks_writes():
    res = execute_sql(_db(), "DELETE FROM customers")
    assert not res.ok
