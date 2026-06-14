"""Schema indexing for retrieval (the "R" in RAG-for-SQL).

Phase 1/2 dumped the *whole* schema into the prompt. That works on Spider's
small databases, but does not scale: a real database can have hundreds of
tables, and a bloated prompt is both expensive and *distracting* to the model.

Here we turn each table into a small, searchable "document":
  * its name and column names, tokenised (snake_case / camelCase split, crude
    singularisation) so a question word like "singers" can match a table "singer",
  * its foreign-key edges, so a retriever can later pull in JOIN "bridge" tables.

This module is pure introspection (no LLM, no network). It is cached per
database so building the index once per run is cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .db import connect_readonly
from .sources import DataSource

# Split "camelCase" boundaries: insert a space between a lower/digit and an upper.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Anything that is not a letter or digit is a token separator.
_NON_WORD = re.compile(r"[^A-Za-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens from an identifier or a question.

    We split snake_case and camelCase, drop very short tokens, and add a crude
    singular form (drop a trailing 's') so "singers" matches "singer". The same
    function is used on both table docs and questions, so the two sides line up.
    """
    text = _CAMEL.sub(" ", text)
    tokens: set[str] = set()
    for part in _NON_WORD.split(text):
        p = part.lower()
        if len(p) >= 2:
            tokens.add(p)
            if len(p) > 3 and p.endswith("s"):
                tokens.add(p[:-1])  # crude singularisation
    return tokens


@dataclass
class TableDoc:
    """One table rendered as a retrievable document."""

    name: str
    columns: list[str]
    fks: list[str] = field(default_factory=list)  # referenced table names
    name_tokens: set[str] = field(default_factory=set)
    column_tokens: set[str] = field(default_factory=set)

    @property
    def text(self) -> str:
        """A natural-language-ish rendering, used for embedding the table."""
        return f"Table {self.name}. Columns: {', '.join(self.columns)}."


@dataclass
class SchemaDoc:
    """All tables of one database, indexed for retrieval."""

    db_id: str
    tables: dict[str, TableDoc]

    def table_names(self) -> list[str]:
        return list(self.tables.keys())


def _build(db_path: str | Path) -> SchemaDoc:
    conn = connect_readonly(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        names = [r[0] for r in cur.fetchall()]

        tables: dict[str, TableDoc] = {}
        for name in names:
            cur.execute(f'PRAGMA table_info("{name}")')
            columns = [r[1] for r in cur.fetchall()]  # r[1] = column name

            cur.execute(f'PRAGMA foreign_key_list("{name}")')
            fks = [r[2] for r in cur.fetchall()]  # r[2] = referenced table

            name_tokens = tokenize(name)
            column_tokens: set[str] = set()
            for col in columns:
                column_tokens |= tokenize(col)

            tables[name] = TableDoc(
                name=name,
                columns=columns,
                fks=fks,
                name_tokens=name_tokens,
                column_tokens=column_tokens,
            )
        return SchemaDoc(db_id=Path(db_path).stem, tables=tables)
    finally:
        conn.close()


@lru_cache(maxsize=128)
def _cached(db_key: str) -> SchemaDoc:
    return _build(db_key)


def load_schema_doc(db_path: str | Path | DataSource) -> SchemaDoc:
    """Return the (cached) retrieval index for a database.

    Phase 9: a non-SQLite `DataSource` is introspected via SQLAlchemy and cached
    by its source id; SQLite (path or DataSource) keeps the original path cache.
    """
    if isinstance(db_path, DataSource) and db_path.kind == "sql":
        return _cached_sql(db_path.id, db_path.url, db_path.dialect)
    if isinstance(db_path, DataSource):
        db_path = db_path.path
    return _cached(str(Path(db_path)))


@lru_cache(maxsize=32)
def _cached_sql(source_id: str, url: str, dialect: str) -> SchemaDoc:
    from .engine import schema_doc_alchemy

    return schema_doc_alchemy(
        DataSource(id=source_id, label=source_id, kind="sql", dialect=dialect, url=url)
    )


def fk_neighbors(schema: SchemaDoc, chosen: set[str]) -> set[str]:
    """Expand a set of tables along foreign keys, one hop, both directions.

    Retrieval by similarity alone can miss the *bridge* table needed to JOIN two
    relevant tables (e.g. an order_items link table whose name shares no words
    with the question). Pulling in FK-connected neighbours keeps JOINs possible.
    """
    extra: set[str] = set()
    for t in chosen:
        doc = schema.tables.get(t)
        if not doc:
            continue
        # outgoing: tables this table references
        for ref in doc.fks:
            if ref in schema.tables:
                extra.add(ref)
    # incoming: tables that reference any chosen table
    for other, doc in schema.tables.items():
        if any(ref in chosen for ref in doc.fks):
            extra.add(other)
    return extra - chosen
