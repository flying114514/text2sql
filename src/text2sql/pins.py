"""Pinned questions / dashboard (Phase 11c): turn one-off Q&A into a habit.

A business user who asks "已完成订单的总金额" once usually wants to see it again
tomorrow. Pinning turns the conversation into a small dashboard: each pin is a
card they re-run with one click to get a *fresh* number.

★ We pin the QUESTION, never the answer. A dashboard of frozen snapshots is worse
  than no dashboard — it quietly goes stale and misleads. So a pin stores the
  question and re-runs it on demand; the last answer is kept only as a timestamped
  preview so a freshly-opened card shows something instantly.

★ A pin is bound to the identity that created it (db_id + governance role). When a
  card refreshes it must re-run AS that role — a viewer's pinned card must not
  silently come back with an analyst's un-masked numbers. So the role travels with
  the pin, not with whatever identity happens to be selected in the switcher. This
  is the P10 governance tie-in.

Refreshing a pin goes through the normal /api/ask pipeline, so it automatically
benefits from the P11a cache (instant within TTL) and the P11b flywheel (learned
examples) — keeping a dashboard up to date is cheap.

Storage: one JSON file data/pins/pins.json, a dict keyed by a hash of
(normalised question, db, role). Pinning the same thing twice updates the existing
card (keeping its created_at) instead of piling up duplicates.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from .config import DATA_DIR, settings

_PINS_PATH = DATA_DIR / "pins" / "pins.json"
_MAX_PINS = 60

_lock = threading.RLock()
_pins: dict[str, dict] | None = None


def _normalize(question: str) -> str:
    """Same answer-preserving folding the cache/feedback use, so 'the same
    question' means the same thing everywhere (and dedups identically)."""
    q = " ".join(question.strip().lower().split())
    return q.rstrip(" 。.?？!！,，;；")


def _key(question: str, db_id: str, role: str | None) -> str:
    raw = "\x00".join([_normalize(question), db_id, role or "-"]).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


# --- persistence ------------------------------------------------------------
def _load() -> dict[str, dict]:
    global _pins
    if _pins is None:
        if _PINS_PATH.exists():
            try:
                _pins = json.loads(_PINS_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _pins = {}
        else:
            _pins = {}
    return _pins


def _save() -> None:
    if _pins is None:
        return
    _PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PINS_PATH.write_text(json.dumps(_pins, ensure_ascii=False, indent=2), encoding="utf-8")


# --- public API -------------------------------------------------------------
def add(
    question: str,
    db_id: str,
    role: str | None = None,
    label: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """Pin a question (or update an existing pin's snapshot). Idempotent per
    (question, db, role): pinning the same thing twice refreshes the preview and
    keeps the original created_at, rather than creating a duplicate card."""
    if not settings.pins_enabled:
        return {"ok": False, "error": "pins disabled"}
    pins = _load()
    k = _key(question, db_id, role)
    now = time.time()
    with _lock:
        existing = pins.get(k) or {}
        pin = {
            "id": k,
            "question": question,
            "label": label or existing.get("label") or question,
            "db_id": db_id,
            "role": role or None,
            "answer": answer if answer is not None else existing.get("answer"),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        pins[k] = pin
        # Cap the board so it stays a curated set, evicting the oldest first.
        if len(pins) > _MAX_PINS:
            oldest = sorted(pins.items(), key=lambda kv: kv[1]["created_at"])
            for kk, _ in oldest[: len(pins) - _MAX_PINS]:
                pins.pop(kk, None)
        _save()
        return {"ok": True, "pin": pin, "total": len(pins)}


def list_pins() -> list[dict]:
    """All pins, oldest-first so cards keep a stable order (new ones append)."""
    pins = _load()
    return sorted(pins.values(), key=lambda p: p.get("created_at", 0.0))


def remove(pin_id: str) -> dict[str, Any]:
    pins = _load()
    with _lock:
        existed = pins.pop(pin_id, None) is not None
        if existed:
            _save()
        return {"ok": existed, "removed": pin_id, "total": len(pins)}


def clear() -> None:
    """Drop every pin (used by tests and a manual reset)."""
    global _pins
    with _lock:
        _pins = {}
        _save()
