"""Phase 0 smoke test #1 — verify the DB execution pipeline (no LLM needed).

Builds the sample database, introspects its schema, and runs a couple of
hand-written queries to prove the read-only execution path works end to end.

    uv run python scripts/smoke_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import make_sample_db  # noqa: E402
from text2sql.db import execute_sql  # noqa: E402
from text2sql.schema import format_schema_for_prompt  # noqa: E402


def main() -> None:
    db_path = make_sample_db.build()
    print(f"[ok] sample db: {db_path}\n")

    print("=== schema (as fed to the LLM) ===")
    print(format_schema_for_prompt(db_path))
    print()

    # 1) A valid query.
    good = "SELECT city, COUNT(*) AS n FROM customers GROUP BY city ORDER BY n DESC"
    res = execute_sql(db_path, good)
    print(f"=== valid query ===\n{good}")
    print(f"ok={res.ok} rows={res.row_count}")
    print(res.preview, "\n")
    assert res.ok and res.row_count > 0, "valid query should succeed"

    # 2) A broken query — we expect ok=False with a clean error (not a crash).
    bad = "SELECT nonexistent_col FROM customers"
    res2 = execute_sql(db_path, bad)
    print(f"=== broken query (expected to fail gracefully) ===\n{bad}")
    print(f"ok={res2.ok} error={res2.error}\n")
    assert not res2.ok and res2.error, "broken query should fail gracefully"

    # 3) A write attempt — read-only mode must block it.
    write = "DELETE FROM customers"
    res3 = execute_sql(db_path, write)
    print(f"=== write attempt (read-only must block) ===\n{write}")
    print(f"ok={res3.ok} error={res3.error}\n")
    assert not res3.ok, "read-only connection must reject writes"

    print("ALL DB SMOKE CHECKS PASSED [OK]")


if __name__ == "__main__":
    main()
