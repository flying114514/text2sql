"""Tests for the execution-accuracy comparison logic (no LLM/network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import result_set_match
from text2sql.models import ExecutionResult


def _ok(rows):
    return ExecutionResult(ok=True, columns=["c"], rows=rows, row_count=len(rows))


def test_order_insensitive_match():
    pred = _ok([(1,), (2,), (3,)])
    gold = _ok([(3,), (1,), (2,)])
    assert result_set_match(pred, gold) is True


def test_order_sensitive_mismatch():
    pred = _ok([(1,), (2,), (3,)])
    gold = _ok([(3,), (2,), (1,)])
    assert result_set_match(pred, gold, order_matters=True) is False


def test_float_rounding():
    pred = _ok([(1.000001,)])
    gold = _ok([(1.0,)])
    assert result_set_match(pred, gold) is True


def test_row_count_mismatch():
    pred = _ok([(1,), (2,)])
    gold = _ok([(1,)])
    assert result_set_match(pred, gold) is False


def test_failed_prediction_never_matches():
    pred = ExecutionResult(ok=False, error="boom")
    gold = _ok([(1,)])
    assert result_set_match(pred, gold) is False
