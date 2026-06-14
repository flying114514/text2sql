"""Peek at the retrieval "documents" the schema index builds for a database.

These docs are NOT files on disk — they are built in memory from the .sqlite at
runtime. This script just renders them so you can *see* what the retriever
actually searches over.

    uv run python scripts/show_schema_doc.py                 # sample.sqlite
    uv run python scripts/show_schema_doc.py data/spider/database/concert_singer/concert_singer.sqlite
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql.schema_index import load_schema_doc  # noqa: E402

db = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "sample.sqlite")
schema = load_schema_doc(db)

print(f"DB: {schema.db_id}   ({len(schema.tables)} tables)\n")
for name, doc in schema.tables.items():
    print(f"== table: {name} ==")
    print(f"  embedding text : {doc.text}")
    print(f"  name tokens    : {sorted(doc.name_tokens)}")
    print(f"  column tokens  : {sorted(doc.column_tokens)}")
    print(f"  foreign keys -> {doc.fks}")
    print()
