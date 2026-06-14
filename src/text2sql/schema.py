"""Schema introspection & prompt formatting.

We extract the real schema from a SQLite file and render it as `CREATE TABLE`
statements. Feeding the model genuine DDL (with types and foreign keys) is far
more reliable than a hand-written summary — and in Phase 3 we will retrieve
only the *relevant* tables instead of dumping the whole schema.
"""

from __future__ import annotations

from pathlib import Path

from .db import connect_readonly
from .sources import DataSource


def get_table_names(db_path: str | Path) -> list[str]:
    conn = connect_readonly(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def get_create_statement(db_path: str | Path, table: str) -> str:
    """Return the original CREATE TABLE statement stored by SQLite."""
    conn = connect_readonly(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
        row = cur.fetchone()
        return (row[0] or "").strip() if row else ""
    finally:
        conn.close()


def sample_rows(db_path: str | Path, table: str, limit: int = 3) -> list[tuple]:
    """A few example rows help the model understand value formats."""
    conn = connect_readonly(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM "{table}" LIMIT {int(limit)}')
        return [tuple(r) for r in cur.fetchall()]
    finally:
        conn.close()


def format_schema_for_prompt(
    db_path: str | Path | DataSource,
    with_samples: bool = True,
    tables: list[str] | None = None,
) -> str:
    """Render the schema as a prompt-ready block.

    Phase 1 dumped every table. From Phase 3 a retriever can pass a `tables`
    subset, so only the relevant DDL reaches the model — cheaper prompts and
    less distraction. `tables` is intersected with the real tables (and order
    preserved) so a stale/hallucinated name can never break rendering.

    Phase 9: a non-SQLite `DataSource` is rendered via the SQLAlchemy backend.
    """
    if isinstance(db_path, DataSource) and db_path.kind == "sql":
        from .engine import format_schema_alchemy

        return format_schema_alchemy(db_path, with_samples=with_samples, tables=tables)
    if isinstance(db_path, DataSource):
        db_path = db_path.path

    all_tables = get_table_names(db_path)
    if tables is not None:
        wanted = set(tables)
        selected = [t for t in all_tables if t in wanted]
    else:
        selected = all_tables

    parts: list[str] = []
    for table in selected:
        ddl = get_create_statement(db_path, table)
        block = ddl if ddl else f"TABLE {table}"
        if with_samples:
            rows = sample_rows(db_path, table)
            if rows:
                block += "\n/* sample rows: " + "; ".join(str(r) for r in rows) + " */"
        parts.append(block)
    return "\n\n".join(parts)
