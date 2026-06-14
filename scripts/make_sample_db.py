"""Create a small but realistic e-commerce SQLite database for local testing.

This gives us a known schema/data to verify the execution pipeline and to demo
the agent before we wire up the full Spider dataset. Run:

    uv run python scripts/make_sample_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.sqlite"

SCHEMA = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT,
    signup_date TEXT
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT,
    price    REAL NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL,
    status      TEXT
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL
);
"""

CUSTOMERS = [
    (1, "Alice", "Beijing", "2024-01-05"),
    (2, "Bob", "Shanghai", "2024-02-11"),
    (3, "Carol", "Beijing", "2024-03-20"),
    (4, "Dave", "Shenzhen", "2024-05-02"),
]

PRODUCTS = [
    (1, "Mechanical Keyboard", "Electronics", 399.0),
    (2, "USB-C Cable", "Accessories", 29.0),
    (3, "Office Chair", "Furniture", 899.0),
    (4, "Desk Lamp", "Furniture", 159.0),
    (5, "Wireless Mouse", "Electronics", 129.0),
]

ORDERS = [
    (1, 1, "2024-04-01", "completed"),
    (2, 1, "2024-04-15", "completed"),
    (3, 2, "2024-04-18", "cancelled"),
    (4, 3, "2024-05-10", "completed"),
    (5, 4, "2024-05-22", "completed"),
]

ORDER_ITEMS = [
    (1, 1, 1, 1),
    (2, 1, 2, 2),
    (3, 2, 5, 1),
    (4, 3, 3, 1),
    (5, 4, 4, 2),
    (6, 4, 2, 3),
    (7, 5, 1, 1),
    (8, 5, 5, 2),
]


def build() -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?)", ORDER_ITEMS)
        conn.commit()
    finally:
        conn.close()
    return DB_PATH


if __name__ == "__main__":
    path = build()
    print(f"Created sample database at: {path}")
