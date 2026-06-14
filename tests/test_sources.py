"""Tests for the Phase 9 multi-source layer.

We don't need a live Postgres to prove the SQLAlchemy backend works: SQLAlchemy
speaks SQLite too, so we point a `kind="sql"` DataSource at the sample database
via a `sqlite:///` URL. That exercises the *generic* path end-to-end —
Inspector-based introspection, schema rendering, read-only execution, and the
dangerous-SQL guard — which is exactly the code real Postgres/MySQL will run.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_sample_db  # noqa: E402
from text2sql import engine, schema_index, sources  # noqa: E402
from text2sql.db import execute_sql  # noqa: E402
from text2sql.schema import format_schema_for_prompt  # noqa: E402
from text2sql.schema_index import load_schema_doc  # noqa: E402


def _sample_sql_source(tmp_path) -> sources.DataSource:
    """A SQLAlchemy DataSource over a *private copy* of the sample DB.

    A SQLAlchemy engine pool keeps the SQLite file open, which on Windows would
    block other tests that rebuild the shared sample.sqlite. Using a per-test
    copy keeps every engine pointed at its own file."""
    if not make_sample_db.DB_PATH.exists():
        make_sample_db.build()
    local = tmp_path / "sample.sqlite"
    shutil.copy(make_sample_db.DB_PATH, local)
    url = f"sqlite:///{local.as_posix()}"
    return sources.DataSource(
        id="sample_sql", label="sample via sqlalchemy", kind="sql", dialect="sqlite", url=url
    )


# --- registry ---------------------------------------------------------------
def test_dialect_of():
    assert sources._dialect_of("postgresql+psycopg2://u:p@h/db") == "postgresql"
    assert sources._dialect_of("mysql+pymysql://u:p@h/db") == "mysql"


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("PG_PASSWORD", "s3cret")
    out = sources._expand_env("postgresql://u:${PG_PASSWORD}@h/db")
    assert out == "postgresql://u:s3cret@h/db"
    # an unset var is left untouched rather than blanked
    assert sources._expand_env("x:${NOPE}") == "x:${NOPE}"


def test_connections_yaml_loaded(tmp_path, monkeypatch):
    cfg = tmp_path / "connections.yaml"
    cfg.write_text(
        "sources:\n"
        "  - id: shop\n"
        "    label: prod\n"
        "    dialect: postgresql\n"
        "    url: postgresql+psycopg2://ro:${PW}@host:5432/shop\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PW", "pw123")
    monkeypatch.setattr(sources, "CONNECTIONS_FILE", cfg)
    ext = sources._load_external()
    assert len(ext) == 1 and ext[0].id == "shop"
    assert ext[0].kind == "sql" and ext[0].dialect == "postgresql"
    assert "pw123" in ext[0].url and "${PW}" not in ext[0].url


# --- SQLAlchemy backend -----------------------------------------------------
def test_alchemy_schema_doc_has_tables_and_fks(tmp_path):
    src = _sample_sql_source(tmp_path)
    doc = engine.schema_doc_alchemy(src)
    assert "customers" in doc.tables and "orders" in doc.tables
    # orders references customers -> the FK edge must be captured for JOINs
    assert "customers" in doc.tables["orders"].fks


def test_alchemy_format_schema_subset(tmp_path):
    src = _sample_sql_source(tmp_path)
    text = format_schema_for_prompt(src, tables=["customers"])
    assert "customers" in text
    assert "CREATE TABLE" in text
    assert "orders" not in text  # subset honoured


def test_alchemy_execute_reads_rows(tmp_path):
    src = _sample_sql_source(tmp_path)
    res = execute_sql(src, "SELECT city, COUNT(*) AS n FROM customers GROUP BY city")
    assert res.ok
    assert "city" in res.columns and "n" in res.columns
    assert res.row_count >= 1


def test_alchemy_guard_blocks_writes(tmp_path):
    """A write must be refused before it ever reaches the database."""
    src = _sample_sql_source(tmp_path)
    res = execute_sql(src, "DELETE FROM customers")
    assert not res.ok
    assert "unsafe" in (res.error or "").lower()


def test_alchemy_bad_sql_returns_error_not_raises(tmp_path):
    src = _sample_sql_source(tmp_path)
    res = execute_sql(src, "SELECT * FROM no_such_table")
    assert not res.ok and res.error  # surfaced for the self-correction loop


def test_load_schema_doc_dispatches_sql_source(tmp_path):
    """The retriever's entry point works on an external source too."""
    src = _sample_sql_source(tmp_path)
    doc = load_schema_doc(src)
    assert isinstance(doc, schema_index.SchemaDoc)
    assert "customers" in doc.tables


# --- JSON-safety of native DB types (the date/Decimal 500 bug) ---------------
def test_jsonable_coerces_native_db_types():
    """Real DBs return date/datetime/Decimal objects; json.dumps rejects them.
    The execution layer must coerce them so the API never 500s."""
    import datetime
    import json
    from decimal import Decimal

    row = {
        "d": datetime.date(2024, 1, 5),
        "ts": datetime.datetime(2024, 1, 5, 13, 30, 0),
        "price": Decimal("399.00"),
        "n": 3,
        "name": "Alice",
        "nothing": None,
    }
    safe = {k: engine._jsonable(v) for k, v in row.items()}
    # must round-trip through the same encoder the API uses
    json.dumps(safe)
    assert safe["d"] == "2024-01-05"
    assert safe["ts"].startswith("2024-01-05T13:30")
    assert safe["price"] == 399.0 and isinstance(safe["price"], float)
    assert safe["n"] == 3 and safe["name"] == "Alice" and safe["nothing"] is None
