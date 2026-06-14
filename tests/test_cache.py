"""Tests for the Phase 11 answer cache.

The cache trades a correctness risk for a big cost/latency win, so the tests
pin down exactly what counts as the *same* question — and, crucially, that the
governance role and the answer-changing knobs are part of the key (serving one
role's cached answer to another would be a data leak).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import cache  # noqa: E402
from text2sql.config import settings  # noqa: E402


def _use_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_CACHE_PATH", tmp_path / "answers.json")
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_ttl_seconds", 86400)
    cache.clear()


KNOBS = {"schema": "lexical", "fewshot": 5, "correct": 0, "analyze": True}


# --- key construction -------------------------------------------------------
def test_normalize_folds_case_whitespace_punctuation():
    assert cache._normalize("  每个城市有多少客户?  ") == "每个城市有多少客户"
    assert cache._normalize("SELECT  x") == cache._normalize("select x")
    assert cache._normalize("有多少订单。") == "有多少订单"


def test_key_is_stable_for_equivalent_questions():
    a = cache.cache_key("有多少客户?", "sample", None, KNOBS)
    b = cache.cache_key("  有多少客户  ", "sample", None, KNOBS)
    assert a == b


def test_key_differs_by_role():
    """The headline P10 tie-in: a viewer and an analyst must never collide."""
    analyst = cache.cache_key("有多少客户", "sample", "analyst", KNOBS)
    viewer = cache.cache_key("有多少客户", "sample", "viewer", KNOBS)
    anon = cache.cache_key("有多少客户", "sample", None, KNOBS)
    assert len({analyst, viewer, anon}) == 3


def test_key_differs_by_db_and_knobs():
    base = cache.cache_key("q", "sample", None, KNOBS)
    assert base != cache.cache_key("q", "shop_pg", None, KNOBS)
    assert base != cache.cache_key("q", "sample", None, {**KNOBS, "fewshot": 0})


# --- store / retrieve -------------------------------------------------------
def test_put_then_get_returns_payload(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    payload = {"ok": True, "answer": "42", "cost_usd": 0.01}
    assert cache.get("有多少客户", "sample", "viewer", KNOBS) is None  # miss
    cache.put("有多少客户", "sample", "viewer", KNOBS, payload)
    hit = cache.get("有多少客户?", "sample", "viewer", KNOBS)  # normalised match
    assert hit is not None and hit["answer"] == "42"


def test_get_returns_a_copy(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    cache.put("q", "sample", None, KNOBS, {"ok": True, "answer": "x"})
    hit = cache.get("q", "sample", None, KNOBS)
    hit["answer"] = "mutated"
    again = cache.get("q", "sample", None, KNOBS)
    assert again["answer"] == "x"  # stored payload untouched


def test_role_isolation_no_cross_serve(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    cache.put("q", "sample", "analyst", KNOBS, {"ok": True, "answer": "Alice"})
    # a viewer asking the same question gets a miss, not the analyst's answer
    assert cache.get("q", "sample", "viewer", KNOBS) is None


def test_ttl_expiry_drops_entry(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "cache_ttl_seconds", 0)
    t = [1000.0]
    monkeypatch.setattr(cache.time, "time", lambda: t[0])
    cache.put("q", "sample", None, KNOBS, {"ok": True})
    t[0] = 1001.0  # one second later, past a 0s TTL
    assert cache.get("q", "sample", None, KNOBS) is None


def test_disabled_is_noop(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "cache_enabled", False)
    cache.put("q", "sample", None, KNOBS, {"ok": True})
    assert cache.get("q", "sample", None, KNOBS) is None


def test_eviction_keeps_cap(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(cache, "_MAX_ENTRIES", 2)
    t = [1000.0]
    monkeypatch.setattr(cache.time, "time", lambda: t[0])
    for i in range(4):
        t[0] += 1
        cache.put(f"q{i}", "sample", None, KNOBS, {"ok": True, "n": i})
    assert cache.stats()["entries"] == 2
    assert cache.get("q0", "sample", None, KNOBS) is None  # oldest evicted
    assert cache.get("q3", "sample", None, KNOBS) is not None  # newest kept


def test_stats_tracks_hits_and_saved_cost(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    cache.put("q", "sample", None, KNOBS, {"ok": True, "cost_usd": 0.002})
    cache.get("q", "sample", None, KNOBS)  # hit
    cache.get("nope", "sample", None, KNOBS)  # miss
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5
    assert abs(s["saved_cost_usd"] - 0.002) < 1e-9
