"""Answer cache (Phase 11): skip the whole LLM pipeline on a repeat question.

A cache hit means we serve a previously-computed answer without calling the
retriever, the model, the database, or the analyst — so a repeated question is
effectively free and instant. That is the cheapest possible win, but only if the
cached answer is *still correct*. Three design decisions keep it honest:

1. Exact (normalised) key, NOT embedding similarity.
   The textbook "semantic cache" embeds the question and serves any cached
   answer whose question is cosine-close. That is unsafe here: "各城市 top 5 客户"
   and "各城市 top 10 客户" embed almost identically but have different answers,
   so a fuzzy hit would silently return wrong data. We normalise the question
   (case / whitespace / trailing punctuation) and require an exact match. An
   embedding layer could sit on top later, but it must never relax correctness.

2. The governance role is part of the key (P10 tie-in).
   Different roles see different rows and columns (RLS + masking), so a cached
   answer is valid only for the identity that produced it. Without role in the
   key, a viewer could be served an analyst's un-masked result — a data leak.

3. Only first-turn questions are cached.
   A follow-up ("那只看北京呢") depends on conversation history, so the question
   string alone does not determine the answer. The service only consults/writes
   the cache when history is empty; this module stays agnostic and the caller
   enforces it.

Entries carry a timestamp and expire after settings.cache_ttl_seconds, so a
changed database eventually stops being served stale. The store is persisted as
one JSON file (survives restarts, makes the cost saving visible across a demo)
and capped to a fixed number of entries (oldest evicted first).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from .config import DATA_DIR, settings

_CACHE_PATH = DATA_DIR / "cache" / "answers.json"
_MAX_ENTRIES = 500

_lock = threading.RLock()
_store: dict[str, dict] | None = None
_stats = {"hits": 0, "misses": 0, "saved_cost_usd": 0.0}


# --- key construction -------------------------------------------------------
def _normalize(question: str) -> str:
    """Fold away differences that should not change the answer: case, internal
    whitespace, and trailing punctuation. Kept deliberately conservative — we
    only collapse things that are obviously answer-preserving."""
    q = " ".join(question.strip().lower().split())
    return q.rstrip(" 。.?？!！,，;；")


def cache_key(question: str, db_id: str, role: str | None, knobs: dict) -> str:
    """A stable hash over everything that can change the answer."""
    parts = [
        _normalize(question),
        db_id,
        role or "-",
        json.dumps(knobs, sort_keys=True, ensure_ascii=False),
    ]
    raw = "\x00".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


# --- persistence ------------------------------------------------------------
def _load() -> dict[str, dict]:
    global _store
    if _store is None:
        if _CACHE_PATH.exists():
            try:
                _store = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _store = {}
        else:
            _store = {}
    return _store


def _save() -> None:
    if _store is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_store, ensure_ascii=False), encoding="utf-8")


# --- public API -------------------------------------------------------------
def get(question: str, db_id: str, role: str | None, knobs: dict) -> dict | None:
    """Return a cached answer payload (a copy) or None. Expired entries are
    dropped. Hits/misses and the cost they saved are accumulated for the UI."""
    if not settings.cache_enabled:
        return None
    store = _load()
    key = cache_key(question, db_id, role, knobs)
    with _lock:
        entry = store.get(key)
        if entry is None:
            _stats["misses"] += 1
            return None
        if time.time() - entry["stored_at"] > settings.cache_ttl_seconds:
            store.pop(key, None)
            _stats["misses"] += 1
            return None
        entry["hits"] += 1
        _stats["hits"] += 1
        _stats["saved_cost_usd"] += entry["payload"].get("cost_usd", 0.0) or 0.0
        return dict(entry["payload"])  # shallow copy so callers can stamp fields


def put(question: str, db_id: str, role: str | None, knobs: dict, payload: dict) -> None:
    """Store a successful answer. Evicts the oldest entries past the cap."""
    if not settings.cache_enabled:
        return
    store = _load()
    key = cache_key(question, db_id, role, knobs)
    with _lock:
        # The (q, db, role) meta lets a 👎 (P11b) invalidate every knob-variant
        # of a question without re-deriving each hashed key.
        store[key] = {
            "payload": payload,
            "stored_at": time.time(),
            "hits": 0,
            "q": _normalize(question),
            "db": db_id,
            "role": role or "-",
        }
        if len(store) > _MAX_ENTRIES:
            for k, _ in sorted(store.items(), key=lambda kv: kv[1]["stored_at"])[
                : len(store) - _MAX_ENTRIES
            ]:
                store.pop(k, None)
        _save()


def invalidate(question: str, db_id: str, role: str | None) -> int:
    """Drop every cached variant of a question for one role (used by a 👎).

    A thumbs-down means the cached answer was wrong, so we must stop serving it.
    We can't re-derive the hashed key (it also depends on the answer-changing
    knobs), so we scan and match on the stored (q, db, role) meta — cheap at our
    cap of a few hundred entries. Returns how many entries were removed."""
    store = _load()
    nq, r = _normalize(question), (role or "-")
    with _lock:
        dead = [
            k
            for k, e in store.items()
            if e.get("q") == nq and e.get("db") == db_id and e.get("role") == r
        ]
        for k in dead:
            store.pop(k, None)
        if dead:
            _save()
        return len(dead)


def stats() -> dict[str, Any]:
    """A small dashboard payload: how much the cache is saving."""
    store = _load()
    hits, misses = _stats["hits"], _stats["misses"]
    total = hits + misses
    return {
        "entries": len(store),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else None,
        "saved_cost_usd": round(_stats["saved_cost_usd"], 6),
    }


def clear() -> None:
    """Drop everything (used by tests and a manual reset)."""
    global _store
    with _lock:
        _store = {}
        _stats.update(hits=0, misses=0, saved_cost_usd=0.0)
        _save()
