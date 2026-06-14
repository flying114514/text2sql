"""SQLite access layer.

Two responsibilities:
  1. execute_sql  — run a query *read-only* and return a typed result.
  2. (schema introspection lives in schema.py, which builds on this.)

Read-only matters: we open the database with SQLite's URI `mode=ro` flag so a
generated query can never mutate or drop data, even before we add explicit
dangerous-statement filtering in a later phase.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .config import settings
from .guard import check_sql_safe
from .models import ExecutionResult
from .sources import DataSource


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode.

    Raises FileNotFoundError early if the path does not exist, because the
    URI-based read-only open would otherwise produce a confusing error.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=10.0, check_same_thread=False)


def execute_sql(
    db_path: str | Path | DataSource,
    sql: str,
    max_rows: int = 1000,
    timeout_s: float | None = None,
) -> ExecutionResult:
    """Execute `sql` read-only and capture results or the error.

    Three defences (Phase 5) wrap the original behaviour:
      * a dangerous-SQL guard rejects non-SELECT / stacked statements *before*
        execution (defence in depth on top of the read-only connection);
      * a watchdog timer interrupts queries that run past `timeout_s`;
      * we still never raise on a bad query — a SQL error is *expected* signal
        the self-correction loop feeds back to the model.

    Phase 9: the target may be a `DataSource`. External (non-SQLite) sources are
    routed to the SQLAlchemy executor; SQLite — whether a bare path or a SQLite
    DataSource — stays on the original sqlite3 path below, unchanged.
    """
    if isinstance(db_path, DataSource) and db_path.kind == "sql":
        from .engine import execute_sql_alchemy  # lazy: keeps SQLite path dep-free

        return execute_sql_alchemy(db_path, sql, max_rows=max_rows, timeout_s=timeout_s)
    if isinstance(db_path, DataSource):
        db_path = db_path.path

    safe, reason = check_sql_safe(sql)
    if not safe:
        return ExecutionResult(ok=False, error=f"blocked unsafe SQL: {reason}", sql=sql)

    timeout_s = settings.db_query_timeout if timeout_s is None else timeout_s

    try:
        conn = connect_readonly(db_path)
    except FileNotFoundError as e:
        return ExecutionResult(ok=False, error=str(e), sql=sql)

    timer: threading.Timer | None = None
    try:
        if timeout_s and timeout_s > 0:
            # interrupt() is safe to call from another thread and aborts a
            # long-running query, which raises OperationalError below.
            timer = threading.Timer(timeout_s, conn.interrupt)
            timer.start()

        cur = conn.cursor()
        cur.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        return ExecutionResult(
            ok=True,
            columns=columns,
            rows=[tuple(r) for r in rows],
            row_count=len(rows),
            sql=sql,
        )
    except sqlite3.Error as e:
        # sqlite3 error messages are concise and informative — exactly what we
        # want to hand back to the model later ("no such column: foo").
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}", sql=sql)
    finally:
        if timer is not None:
            timer.cancel()
        conn.close()
