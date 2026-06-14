"""Data-source registry (Phase 9).

Up to Phase 8 a "database" was always a local SQLite file. Real users have
their data in Postgres / MySQL, so here we introduce a `DataSource`: a named,
typed handle to *some* database, plus a registry that discovers them from two
places:

  * local SQLite files (the sample DB + any downloaded Spider DBs) — unchanged,
    so the whole proven pipeline keeps working exactly as before;
  * external databases declared in `connections.yaml` at the project root
    (gitignored), each with a SQLAlchemy URL.

The point of the abstraction is that the rest of the agent never needs to know
which backend it is talking to: the three leaf functions (load_schema_doc,
format_schema_for_prompt, execute_sql) accept a DataSource and dispatch on its
`kind`. SQLite goes down the original sqlite3 path; everything else goes through
SQLAlchemy (engine.py).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .config import DATA_DIR, ROOT_DIR

CONNECTIONS_FILE = ROOT_DIR / "connections.yaml"
ENV_FILE = ROOT_DIR / ".env"

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """Populate os.environ from the project .env (once).

    pydantic-settings reads .env into the Settings *object* but does not export
    the values to os.environ, so ${VAR} placeholders in connections.yaml — which
    we resolve via os.environ — would otherwise never see secrets kept in .env.
    Real environment variables already set take precedence (setdefault), which
    mirrors pydantic's own precedence rule.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    if not ENV_FILE.exists():
        return
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


@dataclass(frozen=True)
class DataSource:
    """A typed handle to one database.

    `kind` is "sqlite" (local file, original code path) or "sql" (any database
    reachable via a SQLAlchemy URL). `dialect` is the SQLAlchemy dialect name
    ("sqlite" / "postgresql" / "mysql"), which drives read-only + timeout setup.
    """

    id: str
    label: str
    kind: str  # "sqlite" | "sql"
    dialect: str  # "sqlite" | "postgresql" | "mysql" | ...
    path: str = ""  # filesystem path, for kind == "sqlite"
    url: str = ""  # SQLAlchemy URL, for kind == "sql"

    @property
    def is_sqlite_file(self) -> bool:
        return self.kind == "sqlite"


def _expand_env(value: str) -> str:
    """Replace ${VAR} in a connection URL with the environment value.

    Lets `connections.yaml` stay free of secrets (commit-safe) while the actual
    password lives in `.env` / the real environment.
    """
    _ensure_dotenv_loaded()

    def repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))

    return _ENV_REF.sub(repl, value)


def _discover_sqlite() -> list[DataSource]:
    """The original behaviour: the sample DB + every downloaded Spider DB."""
    out: list[DataSource] = []
    sample = DATA_DIR / "sample.sqlite"
    if sample.exists():
        out.append(
            DataSource(
                id="sample",
                label="sample (demo e-commerce)",
                kind="sqlite",
                dialect="sqlite",
                path=str(sample),
            )
        )

    spider_root = DATA_DIR / "spider" / "database"
    if spider_root.exists():
        for db_dir in sorted(spider_root.iterdir()):
            f = db_dir / f"{db_dir.name}.sqlite"
            if f.exists():
                out.append(
                    DataSource(
                        id=db_dir.name,
                        label=f"spider / {db_dir.name}",
                        kind="sqlite",
                        dialect="sqlite",
                        path=str(f),
                    )
                )
    return out


def _dialect_of(url: str) -> str:
    """Extract the dialect from a SQLAlchemy URL ('postgresql+psycopg2://...')."""
    scheme = url.split("://", 1)[0]
    return scheme.split("+", 1)[0].strip().lower()


def _load_external() -> list[DataSource]:
    """Read external databases from connections.yaml (if present)."""
    if not CONNECTIONS_FILE.exists():
        return []
    import yaml  # local import: only needed when a connections file exists

    try:
        data = yaml.safe_load(CONNECTIONS_FILE.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 — a broken file should not crash the app
        raise RuntimeError(f"failed to parse {CONNECTIONS_FILE.name}: {e}") from e

    out: list[DataSource] = []
    for i, entry in enumerate(data.get("sources", []) or []):
        url = _expand_env(str(entry.get("url", "")).strip())
        if not url:
            continue
        sid = str(entry.get("id") or f"db{i}")
        out.append(
            DataSource(
                id=sid,
                label=str(entry.get("label") or sid),
                kind="sql",
                dialect=str(entry.get("dialect") or _dialect_of(url)),
                url=url,
            )
        )
    return out


def list_sources() -> list[DataSource]:
    """All queryable databases: local SQLite first, then external ones."""
    return _discover_sqlite() + _load_external()


def get_source(source_id: str) -> DataSource | None:
    for s in list_sources():
        if s.id == source_id:
            return s
    return None
