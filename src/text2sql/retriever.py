"""Schema retrievers: pick the relevant tables for a question.

Two strategies behind one tiny interface (`select_tables`):

  * LexicalRetriever  — zero-dependency, deterministic, offline. Scores each
    table by token overlap between the question and the table's name/columns
    (table-name hits weighted higher). A strong, explainable baseline.

  * EmbeddingRetriever — embeds each table doc and the question, ranks by cosine
    similarity. Catches semantic matches the lexical version misses
    ("musicians" ~ "singer"). Needs an embedding endpoint; falls back politely.

Both then run **foreign-key expansion** so JOIN bridge tables are not dropped,
and both fall back to the full schema when nothing scores — never silently
returning an empty or wrong table set.

Design choice: we measure these against each other (and against full-schema) in
the eval harness. The retriever is an *interface*, so swapping strategies — or
adding a future hybrid — is a one-line change at the call site.
"""

from __future__ import annotations

import math
from pathlib import Path

from .schema_index import SchemaDoc, fk_neighbors, load_schema_doc, tokenize


class Retriever:
    """Interface: given a question and a database, return the table subset."""

    name = "base"

    def select_tables(self, question: str, db_path: str | Path) -> list[str]:
        raise NotImplementedError


def _finalize(schema: SchemaDoc, scored: list[tuple[str, float]], top_k: int) -> list[str]:
    """Shared selection policy: top-k by score, FK-expand, safe fallback.

    `scored` is (table, score) sorted by score descending.
    """
    best = scored[0][1] if scored else 0.0
    # Nothing matched at all -> safest is the full schema (== baseline for this DB).
    if best <= 0:
        return schema.table_names()

    chosen = {t for t, s in scored[:top_k] if s > 0}
    chosen |= fk_neighbors(schema, chosen)

    # Preserve a stable, schema-declaration order in the prompt.
    return [t for t in schema.table_names() if t in chosen]


class LexicalRetriever(Retriever):
    """Token-overlap retrieval. Table-name matches count more than column hits."""

    name = "lexical"
    NAME_WEIGHT = 3
    COLUMN_WEIGHT = 1

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def _score(self, q_tokens: set[str], doc) -> float:
        s = 0
        for q in q_tokens:
            if q in doc.name_tokens:
                s += self.NAME_WEIGHT
            elif q in doc.column_tokens:
                s += self.COLUMN_WEIGHT
        return float(s)

    def select_tables(self, question: str, db_path: str | Path) -> list[str]:
        schema = load_schema_doc(db_path)
        q_tokens = tokenize(question)
        scored = sorted(
            ((name, self._score(q_tokens, doc)) for name, doc in schema.tables.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return _finalize(schema, scored, self.top_k)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingRetriever(Retriever):
    """Semantic retrieval via cosine similarity over embedded table docs."""

    name = "embed"

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def select_tables(self, question: str, db_path: str | Path) -> list[str]:
        # Imported lazily so the lexical path never requires the embedding deps/config.
        from .embeddings import embed_texts

        schema = load_schema_doc(db_path)
        names = schema.table_names()
        docs = [schema.tables[n].text for n in names]

        # One batched call: the question first, then every table doc.
        vectors = embed_texts([question] + docs)
        q_vec, table_vecs = vectors[0], vectors[1:]

        scored = sorted(
            ((name, _cosine(q_vec, vec)) for name, vec in zip(names, table_vecs, strict=True)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return _finalize(schema, scored, self.top_k)


def get_retriever(kind: str, top_k: int = 5) -> Retriever | None:
    """Factory used by the CLI / eval harness. Returns None for full-schema mode."""
    if kind == "full":
        return None
    if kind == "lexical":
        return LexicalRetriever(top_k=top_k)
    if kind == "embed":
        return EmbeddingRetriever(top_k=top_k)
    raise ValueError(f"unknown retriever kind: {kind!r}")
