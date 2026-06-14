"""Dangerous-SQL guard — a second line of defence (Phase 5).

The read-only connection (db.py) already makes writes physically impossible at
the SQLite level. This guard adds *defence in depth* one step earlier: we parse
the model's SQL with sqlglot and refuse anything that isn't a single read-only
query — DROP/DELETE/UPDATE/INSERT/ALTER/CREATE, or statement stacking
("SELECT ...; DROP ..."). Catching it before execution gives a clear, auditable
"blocked" signal instead of a raw SQLite write error.

Design choice — fail *open* on parse errors: if sqlglot can't parse a query, we
do NOT block it. Some valid SQLite syntax may not parse cleanly, and blocking it
would create false positives; the read-only connection is still there as the
real enforcement. So this layer only blocks what it can *positively* identify as
unsafe.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

# Expression types that mutate data or schema — never allowed.
_WRITE_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
)

# Top-level statement types we consider read-only.
_READ_ROOTS = (exp.Select, exp.Union, exp.With, exp.Subquery)


def check_sql_safe(sql: str) -> tuple[bool, str | None]:
    """Return (is_safe, reason). reason is None when safe."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return True, None  # empty -> let normal execution surface the error

    try:
        statements = [st for st in sqlglot.parse(s, read="sqlite") if st is not None]
    except Exception:
        return True, None  # cannot parse confidently -> defer to read-only conn

    if len(statements) > 1:
        return False, "multiple statements are not allowed"
    if not statements:
        return True, None

    stmt = statements[0]
    for node_type in _WRITE_NODES:
        if next(stmt.find_all(node_type), None) is not None:
            return False, f"{node_type.__name__.upper()} statement is not allowed"

    if not isinstance(stmt, _READ_ROOTS):
        return False, f"only read-only SELECT queries are allowed (got {type(stmt).__name__})"

    return True, None
