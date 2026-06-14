"""Tests for Phase 3 schema retrieval — deterministic, no LLM/network.

We build a tiny throwaway SQLite database (with foreign keys) so the lexical
retriever, FK expansion and the table-recall metric can be exercised in
isolation from any model call.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import gold_tables, table_recall  # noqa: E402
from text2sql.retriever import LexicalRetriever  # noqa: E402
from text2sql.schema_index import fk_neighbors, load_schema_doc, tokenize  # noqa: E402


@pytest.fixture()
def shop_db(tmp_path: Path) -> Path:
    """A 4-table shop schema: customers <- orders <- order_items -> products."""
    db = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE products  (id INTEGER PRIMARY KEY, title TEXT, price REAL);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        """
    )
    conn.commit()
    conn.close()
    return db


def test_tokenize_splits_and_singularizes():
    toks = tokenize("order_items")
    assert "order" in toks and "items" in toks and "item" in toks  # crude singular


def test_tokenize_camelcase():
    assert "customer" in tokenize("customerName") and "name" in tokenize("customerName")


def test_lexical_picks_obvious_table(shop_db: Path):
    selected = LexicalRetriever(top_k=2).select_tables("how many customers per city", shop_db)
    assert "customers" in selected


def test_fk_expansion_pulls_in_bridge_table(shop_db: Path):
    schema = load_schema_doc(shop_db)
    # orders links customers and order_items; expanding it must reach both sides.
    neighbors = fk_neighbors(schema, {"orders"})
    assert "customers" in neighbors  # orders -> customers (outgoing FK)
    assert "order_items" in neighbors  # order_items -> orders (incoming FK)


def test_no_match_falls_back_to_all_tables(shop_db: Path):
    selected = LexicalRetriever(top_k=2).select_tables("zzzz qqqq nonsense", shop_db)
    assert set(selected) == {"customers", "products", "orders", "order_items"}


def test_gold_tables_parses_join():
    sql = "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id"
    assert gold_tables(sql) == {"customers", "orders"}


def test_table_recall_partial():
    # gold needs 2 tables, retrieval found 1 -> 0.5
    assert table_recall(["customers"], "SELECT * FROM customers JOIN orders") == 0.5
    # full-schema mode (None) is not scored
    assert table_recall(None, "SELECT * FROM customers") is None
