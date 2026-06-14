"""Few-shot example retrieval (Phase 4B / "B").

Retrieve a handful of similar "question -> SQL" pairs and show them to the model
as demonstrations. This is RAG again, but the retrieved documents are *examples*
rather than table schemas — a well-known lever for Text2SQL accuracy (the model
picks up SQL idioms and output conventions from nearby examples).

Leakage control (★ important and interview-worthy):
The pool we ship (HF `premai-io/spider` train.json) is NOT database-disjoint from
the dev set — it actually contains all 20 dev databases. Using it naively would
leak: we could retrieve an example on the *same* database (or the identical
question). So `select` always EXCLUDES examples whose db_id equals the query's
db_id, plus any exact-question duplicate. After that filter the demonstrations
are genuinely from other databases — a fair generalisation test.

Speed: we build a token -> example inverted index, so scoring only touches
examples that share a word with the question, not all ~9.7k of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR
from .schema_index import tokenize


@dataclass
class Example:
    db_id: str
    question: str
    query: str
    tokens: set[str] = field(default_factory=set)


def load_example_pool(path: str | Path | None = None) -> list[dict]:
    """Load the few-shot pool (question/query/db_id triples) from train.json."""
    path = Path(path) if path else (DATA_DIR / "spider" / "train.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Few-shot pool not found at {path}.\n"
            "Run: uv run python scripts/prepare_spider.py --train"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"db_id": r["db_id"], "question": r["question"], "query": r["query"]}
        for r in rows
        if r.get("question") and r.get("query")
    ]


class FewShotRetriever:
    """Lexical question-similarity retrieval over a pool of (question, SQL) pairs."""

    name = "fewshot"

    def __init__(self, pool: list[dict], top_k: int = 5) -> None:
        self.top_k = top_k
        self.examples: list[Example] = []
        self.inverted: dict[str, list[int]] = {}
        for item in pool:
            toks = tokenize(item["question"])
            idx = len(self.examples)
            self.examples.append(
                Example(
                    db_id=item["db_id"], question=item["question"], query=item["query"], tokens=toks
                )
            )
            for t in toks:
                self.inverted.setdefault(t, []).append(idx)

    def select(self, question: str, db_id: str | None = None, k: int | None = None) -> list[dict]:
        """Top-k most question-similar examples, excluding the query's own db."""
        k = k or self.top_k
        q_tokens = tokenize(question)

        # Candidate set via inverted index: only examples sharing a token.
        candidates: set[int] = set()
        for t in q_tokens:
            candidates.update(self.inverted.get(t, ()))

        q_lower = question.strip().lower()
        scored: list[tuple[int, int, int]] = []  # (overlap, -len, idx)
        for i in candidates:
            ex = self.examples[i]
            if db_id is not None and ex.db_id == db_id:  # leakage guard
                continue
            if ex.question.strip().lower() == q_lower:  # exact-dup guard
                continue
            overlap = len(q_tokens & ex.tokens)
            if overlap == 0:
                continue
            scored.append((overlap, -len(ex.tokens), i))

        # More overlap first; ties broken toward shorter (more specific) questions.
        scored.sort(reverse=True)
        out: list[dict] = []
        for _overlap, _neg_len, i in scored[:k]:
            ex = self.examples[i]
            out.append({"db_id": ex.db_id, "question": ex.question, "query": ex.query})
        return out


_POOL_CACHE: FewShotRetriever | None = None


def get_fewshot_retriever(top_k: int = 5, path: str | Path | None = None) -> FewShotRetriever:
    """Build (once) and return the few-shot retriever over the train pool."""
    global _POOL_CACHE
    if _POOL_CACHE is None:
        _POOL_CACHE = FewShotRetriever(load_example_pool(path), top_k=top_k)
    _POOL_CACHE.top_k = top_k
    return _POOL_CACHE
