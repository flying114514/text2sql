"""Seed a real Postgres database with the e-commerce sample data (Phase 9).

Why this exists: the SQLAlchemy backend is unit-tested against SQLite-via-URL,
but a *real* Postgres demo is far more convincing on a resume — it proves the
cross-dialect introspection works on genuine Postgres types (SERIAL / DATE /
NUMERIC), and it lets us stand up a least-privilege READ-ONLY role, which is the
real safety net behind the agent's read-only guarantees.

This script is idempotent: it drops and recreates the tables and the read-only
role each run. It connects as the admin (superuser) to do that; the *agent*
then connects only as the read-only role.

Usage (admin URL + the read-only password to provision):
    uv run python scripts/seed_postgres.py \
        --admin-url postgresql+psycopg2://postgres:PASS@localhost:5433/shop \
        --readonly-password ro_pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# reuse the exact sample data so the SQLite and Postgres demos line up
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_sample_db as sample  # noqa: E402

SCHEMA = """
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT,
    signup_date DATE
);

CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT,
    price    NUMERIC(10, 2) NOT NULL
);

CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  DATE NOT NULL,
    status      TEXT
);

CREATE TABLE order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL
);
"""


def _seed_data(conn) -> None:
    conn.execute(
        text(
            "INSERT INTO customers (id, name, city, signup_date) "
            "VALUES (:id, :name, :city, :signup_date)"
        ),
        [
            dict(zip(("id", "name", "city", "signup_date"), r, strict=True))
            for r in sample.CUSTOMERS
        ],
    )
    conn.execute(
        text(
            "INSERT INTO products (id, name, category, price) "
            "VALUES (:id, :name, :category, :price)"
        ),
        [dict(zip(("id", "name", "category", "price"), r, strict=True)) for r in sample.PRODUCTS],
    )
    conn.execute(
        text(
            "INSERT INTO orders (id, customer_id, order_date, status) "
            "VALUES (:id, :customer_id, :order_date, :status)"
        ),
        [
            dict(zip(("id", "customer_id", "order_date", "status"), r, strict=True))
            for r in sample.ORDERS
        ],
    )
    conn.execute(
        text(
            "INSERT INTO order_items (id, order_id, product_id, quantity) "
            "VALUES (:id, :order_id, :product_id, :quantity)"
        ),
        [
            dict(zip(("id", "order_id", "product_id", "quantity"), r, strict=True))
            for r in sample.ORDER_ITEMS
        ],
    )
    # we inserted explicit ids, so bump each SERIAL sequence past them
    for tbl in ("customers", "products", "orders", "order_items"):
        conn.execute(
            text(
                f"SELECT setval(pg_get_serial_sequence('{tbl}', 'id'), "
                f"(SELECT COALESCE(MAX(id), 1) FROM {tbl}))"
            )
        )


def _provision_readonly(conn, db_name: str, password: str) -> None:
    """Create/refresh a least-privilege role the agent will connect as."""
    conn.execute(
        text(
            "DO $$ BEGIN "
            "  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'readonly') THEN "
            "    CREATE ROLE readonly LOGIN; "
            "  END IF; "
            "END $$;"
        )
    )
    conn.execute(text("ALTER ROLE readonly WITH PASSWORD :pw"), {"pw": password})
    conn.execute(text(f'GRANT CONNECT ON DATABASE "{db_name}" TO readonly'))
    conn.execute(text("GRANT USAGE ON SCHEMA public TO readonly"))
    conn.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly"))
    conn.execute(
        text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-url", required=True, help="SQLAlchemy URL with admin rights")
    ap.add_argument(
        "--readonly-password", required=True, help="password to set on the readonly role"
    )
    args = ap.parse_args()

    if not sample.DB_PATH.exists():
        sample.build()  # ensure the sample data module's tables exist (harmless)

    engine = create_engine(args.admin_url, future=True)
    db_name = engine.url.database
    with engine.begin() as conn:
        for stmt in filter(str.strip, SCHEMA.split(";")):
            conn.execute(text(stmt))
        _seed_data(conn)
        _provision_readonly(conn, db_name, args.readonly_password)

    # sanity read-back
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar_one()
        cities = conn.execute(
            text("SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY 2 DESC")
        ).fetchall()
    engine.dispose()
    print(f"Seeded Postgres db '{db_name}': {n} customers; by city = {cities}")
    print("Read-only role 'readonly' provisioned (SELECT-only on public schema).")


if __name__ == "__main__":
    main()
