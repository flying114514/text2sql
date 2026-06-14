"""Feedback flywheel (Phase 11b): turn 👍/👎 into a self-improving agent.

Every answer carries a thumbs-up / thumbs-down. The two signals are NOT
symmetric, and that asymmetry is the whole design:

  👍  The user vouches that this (question -> SQL) is correct *on their own
      database*. That is the highest-value training signal we will ever get, so
      we promote the pair into a verified few-shot pool keyed by db_id and inject
      it ahead of the generic Spider examples on future, similar questions. Ask
      the same kind of question again and the agent has a worked, human-approved
      precedent — it gets more accurate the more it is used (the flywheel).

  👎  The user only tells us the answer was *wrong* — not what the right SQL is.
      A thumbs-down therefore CANNOT become a positive example. Its value is
      different: (a) it is logged for triage (which questions fail), and (b) it
      invalidates that question's answer cache, so we stop serving a result the
      user just rejected. If a previously-👍'd pair later gets 👎, we also remove
      it from the verified pool — we stop teaching something now known to be bad.

★ Same-database is intentional here — the opposite of the Phase 4B leakage guard.
  In *evaluation* we exclude same-db examples to measure generalisation fairly.
  In *production* a verified same-db example is exactly what we want: it is not
  leakage, it is institutional knowledge about this schema. So the verified pool
  (this module) deliberately includes same-db pairs, while the Spider retriever
  (examples.py) deliberately excludes them. They are complementary.

Storage is two files under data/feedback/:
  events.jsonl  — append-only full history (every 👍/👎), for triage/analytics.
  learned.json  — the materialised verified pool: {db_id: {norm_q: {...}}}, the
                  live index the retriever reads (latest 👍 wins; a 👎 removes).
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from . import cache
from .config import DATA_DIR, settings

_EVENTS_PATH = DATA_DIR / "feedback" / "events.jsonl"
_LEARNED_PATH = DATA_DIR / "feedback" / "learned.json"

_lock = threading.RLock()
_learned: dict[str, dict[str, dict]] | None = None

# The Phase 3 schema tokenizer treats every CJK character as a separator, so it
# yields nothing useful on Chinese questions (fine there — the Spider pool is
# English). The flywheel matches the user's real Chinese questions, so we use a
# CJK-aware overlap: ASCII word tokens plus CJK character bigrams, which give a
# meaningful similarity signal without a segmentation dependency.
_ASCII = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def _qtokens(text: str) -> set[str]:
    t = text.lower()
    toks: set[str] = {w for w in _ASCII.findall(t) if len(w) >= 2}
    for run in _CJK.findall(t):
        if len(run) == 1:
            toks.add(run)
        else:
            toks.update(run[i : i + 2] for i in range(len(run) - 1))
    return toks


def _normalize(question: str) -> str:
    """Same answer-preserving folding the cache uses, so 'same question' means
    the same thing in both places."""
    q = " ".join(question.strip().lower().split())
    return q.rstrip(" 。.?？!！,，;；")


# --- persistence ------------------------------------------------------------
def _load() -> dict[str, dict[str, dict]]:
    global _learned
    if _learned is None:
        if _LEARNED_PATH.exists():
            try:
                _learned = json.loads(_LEARNED_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _learned = {}
        else:
            _learned = {}
    return _learned


def _save() -> None:
    if _learned is None:
        return
    _LEARNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LEARNED_PATH.write_text(json.dumps(_learned, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_event(event: dict) -> None:
    _EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# --- public API -------------------------------------------------------------
def record(
    question: str, db_id: str, sql: str, rating: str, role: str | None = None
) -> dict[str, Any]:
    """Record one 👍/👎 and update the flywheel. Returns a small status dict."""
    if not settings.feedback_enabled:
        return {"ok": False, "error": "feedback disabled"}
    if rating not in ("up", "down"):
        return {"ok": False, "error": "rating must be 'up' or 'down'"}

    nq = _normalize(question)
    learned = _load()
    invalidated = 0
    with _lock:
        # Full history first — even rejected/duplicate signals are worth keeping.
        _append_event(
            {
                "ts": time.time(),
                "rating": rating,
                "db_id": db_id,
                "role": role or None,
                "question": question,
                "sql": sql,
            }
        )
        bucket = learned.setdefault(db_id, {})
        if rating == "up":
            if not sql:
                return {"ok": False, "error": "no SQL to learn from"}
            bucket[nq] = {"question": question, "query": sql, "ts": time.time()}
            adopted = True
        else:  # down: never a positive example; drop any stale 👍 and bust cache
            bucket.pop(nq, None)
            adopted = False
            # role here mirrors service.cache_role (None for permissive identities)
            invalidated = cache.invalidate(question, db_id, role)
        _save()
        learned_count = len(bucket)

    return {
        "ok": True,
        "rating": rating,
        "adopted": adopted,
        "learned_for_db": learned_count,
        "cache_invalidated": invalidated,
    }


def learned_examples(db_id: str, question: str, k: int | None = None) -> list[dict]:
    """Top-k verified (question -> SQL) examples for THIS database, ranked by
    lexical overlap with the question. Same-db is the point (see module docstring),
    so unlike the Spider retriever we do not exclude the query's own database."""
    if not settings.feedback_enabled:
        return []
    k = settings.feedback_fewshot_k if k is None else k
    if k <= 0:
        return []
    bucket = _load().get(db_id) or {}
    if not bucket:
        return []
    q_tokens = _qtokens(question)
    scored: list[tuple[int, float, dict]] = []
    for item in bucket.values():
        overlap = len(q_tokens & _qtokens(item["question"]))
        if overlap == 0:
            continue
        # More overlap first; ties broken toward the most recently verified.
        scored.append((overlap, item.get("ts", 0.0), item))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [
        {"db_id": db_id, "question": it["question"], "query": it["query"]}
        for _o, _ts, it in scored[:k]
    ]


def stats() -> dict[str, Any]:
    """Dashboard payload: how much the flywheel has accumulated."""
    learned = _load()
    by_db = {db: len(b) for db, b in learned.items() if b}
    ups = downs = 0
    if _EVENTS_PATH.exists():
        for line in _EVENTS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("rating") == "up":
                ups += 1
            elif e.get("rating") == "down":
                downs += 1
    return {
        "up": ups,
        "down": downs,
        "learned_total": sum(by_db.values()),
        "learned_by_db": by_db,
    }


def clear() -> None:
    """Drop the verified pool (used by tests and a manual reset)."""
    global _learned
    with _lock:
        _learned = {}
        _save()
