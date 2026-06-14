"""Tests for the dangerous-SQL guard (Phase 5) — pure, no LLM/DB."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.guard import check_sql_safe  # noqa: E402


def test_allows_plain_select():
    assert check_sql_safe("SELECT id, name FROM customers WHERE city = 'NY'")[0]


def test_allows_cte_and_join():
    sql = "WITH top AS (SELECT id FROM o LIMIT 5) SELECT * FROM top JOIN c ON c.id = top.id"
    assert check_sql_safe(sql)[0]


def test_allows_union():
    assert check_sql_safe("SELECT 1 UNION SELECT 2")[0]


def test_blocks_delete():
    ok, reason = check_sql_safe("DELETE FROM customers")
    assert not ok and "DELETE" in reason


def test_blocks_drop():
    assert not check_sql_safe("DROP TABLE customers")[0]


def test_blocks_update():
    assert not check_sql_safe("UPDATE customers SET city = 'X'")[0]


def test_blocks_insert():
    assert not check_sql_safe("INSERT INTO customers VALUES (1, 'a', 'b')")[0]


def test_blocks_statement_stacking():
    ok, reason = check_sql_safe("SELECT * FROM customers; DROP TABLE customers")
    assert not ok and "multiple" in reason


def test_blocks_pragma_and_other_commands():
    # PRAGMA parses as a non-SELECT root -> blocked by the top-level check.
    assert not check_sql_safe("PRAGMA table_info(customers)")[0]


def test_empty_defers():
    assert check_sql_safe("")[0]
