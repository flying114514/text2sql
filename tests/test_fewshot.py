"""Tests for Phase 4B few-shot retrieval — deterministic, no LLM/network.

Focus on the two things that matter: (1) the leakage guard never returns an
example from the query's own database, and (2) more question-similar examples
rank higher.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql.examples import FewShotRetriever  # noqa: E402

POOL = [
    {
        "db_id": "db1",
        "question": "how many singers are there",
        "query": "SELECT count(*) FROM singer",
    },
    {
        "db_id": "db2",
        "question": "how many singers do we have",
        "query": "SELECT count(*) FROM singers",
    },
    {
        "db_id": "db3",
        "question": "how many singers are there really now",
        "query": "SELECT count(*) FROM s",
    },
    {"db_id": "db2", "question": "list all the cities by area", "query": "SELECT city FROM geo"},
]


def test_excludes_same_database_leakage():
    r = FewShotRetriever(POOL, top_k=5)
    out = r.select("how many singers are there", db_id="db1", k=5)
    # db1 contains the identical question — must never be returned for a db1 query.
    assert all(ex["db_id"] != "db1" for ex in out)


def test_more_similar_ranks_first():
    r = FewShotRetriever(POOL, top_k=5)
    out = r.select("how many singers are there", db_id="db1", k=5)
    # db3's "...are there really now" shares the most words -> should rank first.
    assert out[0]["db_id"] == "db3"


def test_respects_k_and_drops_zero_overlap():
    r = FewShotRetriever(POOL, top_k=5)
    out = r.select("how many singers are there", db_id="db1", k=1)
    assert len(out) == 1
    # The unrelated "list all the cities" example shares no words -> never selected.
    assert all("cities" not in ex["question"] for ex in r.select("singers count", db_id="db1", k=5))


def test_exact_question_excluded_even_across_dbs():
    pool = [{"db_id": "dbX", "question": "how many singers are there", "query": "SELECT 1"}]
    r = FewShotRetriever(pool, top_k=5)
    out = r.select("how many singers are there", db_id="dbQ", k=5)
    assert out == []  # exact-duplicate question is filtered out
