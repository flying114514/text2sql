"""SQLAlchemy backend (Phase 9): talk to *any* database, read-only.

This is the non-SQLite half of the data layer. Given a `DataSource` with a
SQLAlchemy URL, it provides the same three capabilities the SQLite path does,
so the rest of the agent is backend-agnostic:

  * `schema_doc_alchemy`     — build the retrieval index (SchemaDoc) via the
    dialect-agnostic Inspector (tables, columns, foreign keys).
  * `format_schema_alchemy`  — render a CREATE-TABLE-ish schema block for the
    prompt, with a few sample rows.
  * `execute_sql_alchemy`    — run a SELECT read-only and return an
    ExecutionResult, identical in shape to the SQLite executor.

Reliability — defence in depth, on top of the sqlglot guard that already runs
before any execution:
  1. a read-only transaction (`SET TRANSACTION READ ONLY` on PG/MySQL) so even
     a query that slips past the guard cannot mutate data;
  2. a per-statement timeout set in the database itself (statement_timeout /
     max_execution_time) — far more reliable than a client-side watchdog;
  3. we always ROLLBACK — nothing this agent does is ever committed.

The real production answer is still a dedicated **read-only role**; the above is
belt-and-suspenders so a misconfigured account cannot do damage.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import uuid
from functools import lru_cache

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import settings
from .guard import check_sql_safe
from .models import ExecutionResult
from .schema_index import SchemaDoc, TableDoc, tokenize
from .sources import DataSource


@lru_cache(maxsize=32)
def get_engine(url: str) -> Engine:
    """One pooled engine per URL (cached). `pool_pre_ping` survives idle drops."""
    return create_engine(url, pool_pre_ping=True, future=True)


def _jsonable(value):
    """Coerce a DB cell into something json.dumps can handle.

    SQLite stored everything as text/number, so its rows were always JSON-safe.
    Real databases return native Python objects — datetime.date / datetime,
    decimal.Decimal, UUID, bytes — which the JSON encoder rejects. We normalise
    at the execution boundary so the whole pipeline (API, chart, analyst) keeps
    treating rows as simple values. Decimals become float (chart-friendly);
    dates/times become ISO strings (how SQLite presented them anyway)."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (list, tuple)):  # e.g. Postgres array columns
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):  # e.g. JSON/JSONB columns
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)  # last resort: never let serialization 500 the request


# --- read-only + timeout per dialect ---------------------------------------
def _harden_connection(conn, dialect: str, timeout_s: float) -> None:
    """Make this connection read-only and time-bounded, best-effort per dialect.

    Each statement is wrapped in try/except: a dialect that doesn't support one
    of these knobs must not break the actual query (the sqlglot guard is the
    primary write-protection regardless)."""
    ms = int(max(0.0, timeout_s) * 1000)
    stmts: list[str] = []
    if dialect == "postgresql":
        stmts = ["SET TRANSACTION READ ONLY"]
        if ms:
            stmts.append(f"SET LOCAL statement_timeout = {ms}")
    elif dialect == "mysql":
        stmts = ["SET SESSION TRANSACTION READ ONLY"]
        if ms:
            stmts.append(f"SET SESSION max_execution_time = {ms}")
    # sqlite (or unknown) via SQLAlchemy: nothing to set; guard covers writes.
    for s in stmts:
        try:
            conn.execute(text(s))
        except SQLAlchemyError:
            pass


# --- introspection ----------------------------------------------------------
def schema_doc_alchemy(source: DataSource) -> SchemaDoc:
    """Build the retrieval index from a live database via the Inspector."""
    insp = inspect(get_engine(source.url))
    tables: dict[str, TableDoc] = {}
    for name in insp.get_table_names():
        columns = [c["name"] for c in insp.get_columns(name)]
        fks = [
            fk["referred_table"] for fk in insp.get_foreign_keys(name) if fk.get("referred_table")
        ]
        column_tokens: set[str] = set()
        for col in columns:
            column_tokens |= tokenize(col)
        tables[name] = TableDoc(
            name=name,
            columns=columns,
            fks=fks,
            name_tokens=tokenize(name),
            column_tokens=column_tokens,
        )
    return SchemaDoc(db_id=source.id, tables=tables)


def _render_create(insp, table: str) -> str:
    """A readable CREATE-TABLE-ish block; we don't need exact DDL, just enough
    type and key information for the model to write correct SQL."""
    cols = insp.get_columns(table)
    pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
    lines = []
    for c in cols:
        parts = [f'  "{c["name"]}"', str(c.get("type", ""))]
        if c["name"] in pk:
            parts.append("PRIMARY KEY")
        if not c.get("nullable", True):
            parts.append("NOT NULL")
        lines.append(" ".join(p for p in parts if p).rstrip())
    for fk in insp.get_foreign_keys(table):
        ref_t = fk.get("referred_table")
        cc = ", ".join(fk.get("constrained_columns") or [])
        rc = ", ".join(fk.get("referred_columns") or [])
        if ref_t and cc:
            lines.append(f'  FOREIGN KEY ({cc}) REFERENCES "{ref_t}" ({rc})')
    return f'CREATE TABLE "{table}" (\n' + ",\n".join(lines) + "\n)"


def format_schema_alchemy(
    source: DataSource,
    *,
    with_samples: bool = True,
    tables: list[str] | None = None,
) -> str:
    """Render the schema (optionally a subset) as a prompt-ready block."""
    eng = get_engine(source.url)
    insp = inspect(eng)
    all_tables = insp.get_table_names()
    if tables is not None:
        wanted = set(tables)
        selected = [t for t in all_tables if t in wanted]
    else:
        selected = all_tables

    parts: list[str] = []
    for table in selected:
        block = _render_create(insp, table)
        if with_samples:
            try:
                with eng.connect() as conn:
                    _harden_connection(conn, source.dialect, settings.db_query_timeout)
                    rows = conn.execute(text(f'SELECT * FROM "{table}" LIMIT 3')).fetchall()
                    conn.rollback()
                if rows:
                    block += "\n/* sample rows: " + "; ".join(str(tuple(r)) for r in rows) + " */"
            except SQLAlchemyError:
                pass  # sampling is best-effort; never block schema rendering
        parts.append(block)
    return "\n\n".join(parts)


# --- execution --------------------------------------------------------------
def execute_sql_alchemy(
    source: DataSource,
    sql: str,
    max_rows: int = 1000,
    timeout_s: float | None = None,
) -> ExecutionResult:
    """Run `sql` read-only against an external database and capture the result.

    Mirrors db.execute_sql: the dangerous-SQL guard runs first, errors are
    returned (not raised) so the self-correction loop can feed them back."""
    safe, reason = check_sql_safe(sql)
    if not safe:
        return ExecutionResult(ok=False, error=f"blocked unsafe SQL: {reason}", sql=sql)

    timeout_s = settings.db_query_timeout if timeout_s is None else timeout_s
    try:
        eng = get_engine(source.url)
        with eng.connect() as conn:
            _harden_connection(conn, source.dialect, timeout_s)
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchmany(max_rows)
            conn.rollback()  # never commit anything
        return ExecutionResult(
            ok=True,
            columns=columns,
            rows=[tuple(_jsonable(v) for v in r) for r in rows],
            row_count=len(rows),
            sql=sql,
        )
    except SQLAlchemyError as e:
        # Surface the database's own message — that's what the model repairs against.
        msg = str(getattr(e, "orig", e)).strip() or type(e).__name__
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {msg}", sql=sql)
