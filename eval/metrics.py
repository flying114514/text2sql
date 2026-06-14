"""Evaluation metrics — the heart of "execution accuracy".

The idea (the standard way Text2SQL is judged): we don't compare SQL *strings*
(there are many correct ways to write the same query). Instead we **execute**
both the predicted SQL and the gold SQL against the same database and compare
their result sets.

Simplifications we make (and document honestly, like a real eval would):
  * Rows are compared as a multiset (order-insensitive) unless the case is
    explicitly marked order-sensitive — many questions don't imply an order.
  * Floats are rounded to 4 decimals before comparison to avoid noise.
The official Spider "test-suite" metric is stricter; ours is a reasonable,
defensible approximation that is transparent about its assumptions.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from text2sql.models import ExecutionResult


def _norm_value(v: object) -> object:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, bool):  # bool is a subclass of int — keep it stable
        return int(v)
    return v


def _norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_value(v) for v in row) for row in rows]


def result_set_match(
    pred: ExecutionResult,
    gold: ExecutionResult,
    *,
    order_matters: bool = False,
) -> bool:
    """True if the predicted query's results match the gold query's results."""
    # A prediction that failed to execute can never match.
    if not pred.ok or not gold.ok:
        return False

    p = _norm_rows(pred.rows)
    g = _norm_rows(gold.rows)

    if len(p) != len(g):
        return False

    if order_matters:
        return p == g

    # Order-insensitive multiset comparison.
    return sorted(p, key=lambda r: tuple(str(x) for x in r)) == sorted(
        g, key=lambda r: tuple(str(x) for x in r)
    )


def gold_tables(sql: str) -> set[str]:
    """Extract the (lowercased) table names a SQL query reads from.

    Used to score the *retriever* independently of the LLM: if the gold query
    touches tables {a, b}, did retrieval put both in the prompt? We parse with
    sqlglot rather than regex so aliases, subqueries and JOINs are handled
    properly. On a parse failure we return an empty set (counted as no demand).
    """
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    names: set[str] = set()
    for tbl in tree.find_all(exp.Table):
        if tbl.name:
            names.add(tbl.name.lower())
    return names


def table_recall(selected: list[str] | None, gold_sql: str) -> float | None:
    """Fraction of the gold query's tables that retrieval included.

    Returns None when retrieval wasn't used (full-schema mode) or when the gold
    query references no parseable table, so those cases don't skew the average.
    """
    if selected is None:
        return None
    gold = gold_tables(gold_sql)
    if not gold:
        return None
    chosen = {t.lower() for t in selected}
    return len(gold & chosen) / len(gold)
