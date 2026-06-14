"""Tests for the Phase 11c pinned-questions dashboard.

The two design rules are what the tests pin down: a pin is keyed by
(question, db, role) so it dedups and is governance-scoped, and re-pinning the
same question updates the snapshot in place instead of piling up duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import pins  # noqa: E402
from text2sql.config import settings  # noqa: E402


def _use_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "_PINS_PATH", tmp_path / "pins.json")
    monkeypatch.setattr(settings, "pins_enabled", True)
    pins.clear()


def test_add_creates_a_pin(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    res = pins.add("已完成订单的总金额", "sample", answer="1648 元")
    assert res["ok"] and res["total"] == 1
    board = pins.list_pins()
    assert len(board) == 1
    assert board[0]["question"] == "已完成订单的总金额"
    assert board[0]["answer"] == "1648 元"
    assert board[0]["id"] == res["pin"]["id"]


def test_repin_same_question_updates_in_place(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    first = pins.add("总订单数", "sample", answer="100")["pin"]
    # a later refresh re-pins the same question with a fresh snapshot
    again = pins.add("总订单数?", "sample", answer="105")["pin"]  # normalised match
    assert len(pins.list_pins()) == 1  # not duplicated
    assert again["id"] == first["id"]
    assert again["created_at"] == first["created_at"]  # original creation kept
    assert pins.list_pins()[0]["answer"] == "105"  # snapshot refreshed


def test_pins_are_scoped_by_role(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    pins.add("各城市客户数", "sample", role="viewer", answer="v")
    pins.add("各城市客户数", "sample", role="analyst", answer="a")
    # same question, two identities -> two distinct cards (governance scope)
    assert len(pins.list_pins()) == 2


def test_pins_are_scoped_by_db(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    pins.add("有多少订单", "sample", answer="s")
    pins.add("有多少订单", "shop_pg", answer="p")
    assert len(pins.list_pins()) == 2


def test_list_is_oldest_first(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    a = pins.add("问题一", "sample")["pin"]
    b = pins.add("问题二", "sample")["pin"]
    ids = [p["id"] for p in pins.list_pins()]
    assert ids == [a["id"], b["id"]]  # stable append order


def test_remove(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    pid = pins.add("待删除", "sample")["pin"]["id"]
    assert pins.remove(pid)["ok"] is True
    assert pins.list_pins() == []
    assert pins.remove(pid)["ok"] is False  # idempotent


def test_disabled_is_noop(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "pins_enabled", False)
    assert pins.add("q", "sample")["ok"] is False
    assert pins.list_pins() == []
