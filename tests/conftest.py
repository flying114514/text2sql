"""Shared test fixtures.

The Phase 11 answer cache persists to data/cache/answers.json and is on by
default, which would let one test's result be served to another (and let a
second suite run hit a cache the first run populated — e.g. breaking the exact
token-count assertions in test_service.py). So we disable it for every test by
default and point its file at a throwaway path. Tests that exercise the cache
itself (test_cache.py) re-enable it against their own tmp file via monkeypatch.

The Phase 11b feedback flywheel (feedback.py) likewise persists to
data/feedback/ and is read on every answer_query() call (learned examples). We
redirect its files to a tmp path so tests neither pollute nor read real data.
The Phase 11c pins (pins.py) persist to data/pins/; redirected the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import cache, feedback, pins  # noqa: E402
from text2sql.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(cache, "_CACHE_PATH", tmp_path / "answers.json")
    monkeypatch.setattr(feedback, "_EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(feedback, "_LEARNED_PATH", tmp_path / "learned.json")
    monkeypatch.setattr(pins, "_PINS_PATH", tmp_path / "pins.json")
    cache.clear()
    feedback.clear()
    pins.clear()
    yield
