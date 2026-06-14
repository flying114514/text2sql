"""Tests for the Phase 11b feedback flywheel.

The asymmetry between 👍 and 👎 is the whole design, so the tests pin it down:
👍 adopts a verified *same-db* example (the opposite of the Phase 4B leakage
guard); 👎 never becomes a positive example, removes any stale 👍, and busts the
answer cache for that question.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import cache, feedback  # noqa: E402
from text2sql.config import settings  # noqa: E402


def _use_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(feedback, "_LEARNED_PATH", tmp_path / "learned.json")
    monkeypatch.setattr(settings, "feedback_enabled", True)
    feedback.clear()


KNOBS = {"schema": "lexical", "fewshot": 5, "correct": 0, "analyze": True}


# --- 👍 adopts a verified example ------------------------------------------
def test_thumbs_up_adopts_same_db_example(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    res = feedback.record(
        "每个城市有多少客户", "sample", "SELECT city, COUNT(*) FROM customers GROUP BY city", "up"
    )
    assert res["ok"] and res["adopted"] and res["learned_for_db"] == 1
    # ...and it is retrievable for the SAME database (leakage guard inverted).
    ex = feedback.learned_examples("sample", "每个城市的客户数量", k=3)
    assert len(ex) == 1
    assert ex[0]["db_id"] == "sample"
    assert ex[0]["query"].startswith("SELECT city")


def test_learned_examples_are_db_scoped(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    feedback.record("有多少客户", "sample", "SELECT COUNT(*) FROM customers", "up")
    # a different database never sees another db's verified examples
    assert feedback.learned_examples("shop_pg", "有多少客户", k=3) == []
    assert len(feedback.learned_examples("sample", "有多少客户", k=3)) == 1


def test_learned_examples_ranked_by_overlap_and_capped(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    feedback.record("每个城市有多少客户", "sample", "SELECT a", "up")
    feedback.record("每个城市有多少订单", "sample", "SELECT b", "up")
    feedback.record("价格最高的产品", "sample", "SELECT c", "up")
    got = feedback.learned_examples("sample", "每个城市的客户数量", k=2)
    assert len(got) == 2  # capped at k, the no-overlap one drops
    assert got[0]["query"] == "SELECT a"  # most shared characters first


# --- 👎 is not the inverse of 👍 -------------------------------------------
def test_thumbs_down_is_not_a_positive_example(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    res = feedback.record("有多少客户", "sample", "SELECT wrong", "down")
    assert res["ok"] and res["adopted"] is False
    assert feedback.learned_examples("sample", "有多少客户", k=3) == []


def test_thumbs_down_removes_a_prior_thumbs_up(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    feedback.record("有多少客户", "sample", "SELECT COUNT(*) FROM customers", "up")
    assert feedback.learned_examples("sample", "有多少客户", k=3)  # adopted
    feedback.record("有多少客户?", "sample", "", "down")  # normalised match
    assert feedback.learned_examples("sample", "有多少客户", k=3) == []  # un-adopted


def test_thumbs_down_invalidates_cache(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    # arm the cache for this question/role
    monkeypatch.setattr(cache, "_CACHE_PATH", tmp_path / "answers.json")
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_ttl_seconds", 86400)
    cache.clear()
    cache.put("有多少客户", "sample", "viewer", KNOBS, {"ok": True, "answer": "stale"})
    assert cache.get("有多少客户", "sample", "viewer", KNOBS) is not None
    # a 👎 from that same role must drop every knob-variant of the question
    res = feedback.record("有多少客户", "sample", "SELECT x", "down", role="viewer")
    assert res["cache_invalidated"] >= 1
    assert cache.get("有多少客户", "sample", "viewer", KNOBS) is None


# --- misc -------------------------------------------------------------------
def test_record_rejects_bad_rating(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    assert feedback.record("q", "sample", "SELECT 1", "meh")["ok"] is False


def test_stats_counts_up_down_and_pool(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    feedback.record("q1", "sample", "SELECT 1", "up")
    feedback.record("q2", "sample", "SELECT 2", "up")
    feedback.record("q3", "sample", "SELECT 3", "down")
    s = feedback.stats()
    assert s["up"] == 2 and s["down"] == 1
    assert s["learned_total"] == 2
    assert s["learned_by_db"]["sample"] == 2


def test_disabled_is_noop(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "feedback_enabled", False)
    assert feedback.record("q", "sample", "SELECT 1", "up")["ok"] is False
    assert feedback.learned_examples("sample", "q", k=3) == []
