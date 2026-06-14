"""Data governance (Phase 10): authn-lite + row/column security + PII masking + audit.

Enforced at the EXECUTION boundary, never in the prompt. An LLM can be talked out
of any instruction it is given, so access control must be deterministic code the
model cannot influence. Per query the service runs:

  resolve_principal(role|api_key, source) -> Principal       # who is asking
  enforce(sql, principal, source)         -> Enforcement     # block / rewrite SQL
       · denied column referenced  -> refuse (like a DB "permission denied")
       · row filters               -> rewrite via a security-barrier subquery
  ... execute the (possibly rewritten) SQL ...
  redact(cols, rows, principal, source, sql) -> mask PII / drop denied columns
  audit(...)                                  -> append an immutable access record

Two clean layers, two files:
  * semantics/<id>.yaml `pii:`  = WHAT is sensitive   (data classification)
  * policies.yaml               = WHO may see it / how (access policy)

Fail direction matters and is the opposite of the dangerous-SQL guard:
the guard fails OPEN (the read-only connection is the real backstop), but row/
column security fails CLOSED — if we cannot parse a query for a restricted
principal we refuse it, because guessing would leak rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

import sqlglot
from sqlglot import exp

from .config import DATA_DIR, ROOT_DIR
from .semantics import load_semantics

POLICIES_FILE = ROOT_DIR / "policies.yaml"
AUDIT_DIR = DATA_DIR / "audit"

# SQLAlchemy dialect name -> sqlglot dialect name.
_SQLGLOT_DIALECT = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
}


# --- principal & enforcement models ----------------------------------------
@dataclass
class Principal:
    """Who is asking, already scoped to ONE data source.

    `permissive` means governance is effectively off (no policies.yaml) — every
    check short-circuits so the pipeline behaves exactly as it did pre-P10.
    """

    role: str
    label: str
    can_see_pii: bool = False
    row_filters: dict = field(default_factory=dict)  # {table: predicate}
    deny_columns: set = field(default_factory=set)  # {"customers.signup_date"}
    permissive: bool = False

    @property
    def restricts_rows_or_cols(self) -> bool:
        return bool(self.row_filters or self.deny_columns)


@dataclass
class Enforcement:
    """Result of pre-execution enforcement on a SQL string."""

    allowed: bool
    sql: str  # rewritten (RLS) or original
    reason: str | None = None  # set when allowed is False
    row_security: list[str] = field(default_factory=list)  # ["customers: city='Beijing'"]
    blocked_columns: list[str] = field(default_factory=list)


# --- policy loading ---------------------------------------------------------
@lru_cache(maxsize=1)
def load_policies() -> dict:
    """Parse policies.yaml, or {} if there is none (governance disabled)."""
    if not POLICIES_FILE.exists():
        return {}
    import yaml  # local import: only needed when a policies file exists

    try:
        return yaml.safe_load(POLICIES_FILE.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 — a broken policy file must not crash answering
        raise RuntimeError(f"failed to parse {POLICIES_FILE.name}: {e}") from e


def list_roles() -> list[dict]:
    """Roles the UI can offer in its identity switcher ([] when governance off)."""
    pol = load_policies()
    out = []
    for name, rdef in (pol.get("roles") or {}).items():
        rdef = rdef or {}
        out.append(
            {
                "role": name,
                "label": str(rdef.get("label") or name),
                "can_see_pii": bool(rdef.get("can_see_pii", False)),
            }
        )
    return out


def resolve_principal(role: str | None, api_key: str | None, source_id: str) -> Principal:
    """Map a requested role / API key to a Principal scoped to `source_id`.

    Precedence: explicit valid role > api_key mapping > default_role. Unknown
    identities fall to the least-privileged default (deny by default).
    """
    pol = load_policies()
    if not pol:
        return Principal(role="admin", label="管理员(无策略)", can_see_pii=True, permissive=True)

    roles = pol.get("roles") or {}
    chosen = None
    if role and role in roles:
        chosen = role
    elif api_key and api_key in (pol.get("identities") or {}):
        chosen = pol["identities"][api_key]
    if not chosen:
        chosen = pol.get("default_role") or (next(iter(roles), None))

    rdef = (roles.get(chosen) or {}) if chosen else {}
    return Principal(
        role=chosen or "unknown",
        label=str(rdef.get("label") or chosen or "未知"),
        can_see_pii=bool(rdef.get("can_see_pii", False)),
        row_filters=dict((rdef.get("row_filters") or {}).get(source_id) or {}),
        deny_columns={
            str(c).strip() for c in ((rdef.get("deny_columns") or {}).get(source_id) or [])
        },
    )


# --- enforcement (pre-execution) -------------------------------------------
def _dialect(source) -> str | None:
    return _SQLGLOT_DIALECT.get(getattr(source, "dialect", ""), None)


def _bare(name: str) -> str:
    """Last dotted segment, lowercased: 'customers.signup_date' -> 'signup_date'."""
    return name.split(".")[-1].strip().lower()


def enforce(sql: str, principal: Principal, source) -> Enforcement:
    """Apply column denial + row-level security to `sql` before execution."""
    if principal.permissive or not principal.restricts_rows_or_cols:
        return Enforcement(allowed=True, sql=sql)

    dialect = _dialect(source)
    try:
        ast = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001
        # Fail CLOSED: a restricted principal must never run an unparseable query.
        return Enforcement(
            allowed=False, sql=sql, reason="无法解析该 SQL,出于行/列安全已拒绝执行(数据治理)"
        )

    # Column denial — refuse if a forbidden column is referenced anywhere.
    if principal.deny_columns:
        denied_bare = {_bare(c) for c in principal.deny_columns}
        hit = sorted({c.name for c in ast.find_all(exp.Column) if c.name.lower() in denied_bare})
        if hit:
            return Enforcement(
                allowed=False,
                sql=sql,
                blocked_columns=hit,
                reason=f"角色「{principal.label}」无权访问字段:{'、'.join(hit)}",
            )

    # Row-level security — swap each governed table for a security-barrier view.
    applied: list[str] = []
    if principal.row_filters:
        targets = [t for t in ast.find_all(exp.Table) if t.name in principal.row_filters]
        for t in targets:
            pred = principal.row_filters[t.name]
            inner = sqlglot.parse_one(f"SELECT * FROM {t.name} WHERE {pred}", read=dialect)
            t.replace(inner.subquery(alias=(t.alias or t.name)))
            applied.append(f"{t.name}: {pred}")
        if applied:
            sql = ast.sql(dialect=dialect)

    return Enforcement(allowed=True, sql=sql, row_security=sorted(set(applied)))


# --- redaction (post-execution) --------------------------------------------
def _mask_value(v):
    """Reveal only the first character: 'Alice' -> 'A****', '张三' -> '张*'."""
    if v is None:
        return None
    s = str(v)
    if not s:
        return s
    if len(s) == 1:
        return "*"
    return s[0] + "*" * min(len(s) - 1, 4)


def _projection_sources(sql: str, dialect: str | None) -> list[str | None]:
    """Per output column, the underlying source column name when it is a simple
    column reference (so `name AS n` still maps to 'name'); None otherwise."""
    try:
        ast = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # noqa: BLE001
        return []
    sel = ast if isinstance(ast, exp.Select) else ast.find(exp.Select)
    if sel is None:
        return []
    out: list[str | None] = []
    for proj in sel.expressions:
        target = proj.this if isinstance(proj, exp.Alias) else proj
        if isinstance(target, exp.Column):
            out.append(target.name)
        elif isinstance(target, exp.Star):
            out.append("*")
        else:
            out.append(None)
    return out


def redact(
    columns: list[str], rows: list[tuple], principal: Principal, source, sql: str
) -> tuple[list[str], list[tuple], list[str], list[str]]:
    """Mask PII the principal may not see and drop denied columns from output.

    Returns (columns, rows, masked_columns, dropped_columns). Denial blocks at
    enforce() when a column is named explicitly; this drop is the backstop for
    `SELECT *`, which enforce() cannot inspect without the schema.
    """
    if principal.permissive:
        return columns, rows, [], []

    pii_bare: set[str] = set()
    if not principal.can_see_pii:
        layer = load_semantics(source.id)
        if layer:
            pii_bare = {_bare(p) for p in layer.pii}
    denied_bare = {_bare(c) for c in principal.deny_columns}
    if not pii_bare and not denied_bare:
        return columns, rows, [], []

    proj = _projection_sources(sql, _dialect(source))

    def hits(label: str, src: str | None, bareset: set[str]) -> bool:
        return label in bareset or (src in bareset if src else False)

    keep_idx: list[int] = []
    mask_idx: set[int] = set()
    masked: list[str] = []
    dropped: list[str] = []
    for i, col in enumerate(columns):
        label = str(col).lower()
        src = proj[i].lower() if i < len(proj) and proj[i] and proj[i] != "*" else None

        if denied_bare and hits(label, src, denied_bare):
            dropped.append(col)
            continue
        keep_idx.append(i)
        if pii_bare and hits(label, src, pii_bare):
            mask_idx.add(i)
            masked.append(col)

    if not dropped and not mask_idx:
        return columns, rows, [], []

    new_cols = [columns[i] for i in keep_idx]
    new_rows = [tuple(_mask_value(r[i]) if i in mask_idx else r[i] for i in keep_idx) for r in rows]
    return new_cols, new_rows, masked, dropped


# --- audit ------------------------------------------------------------------
def audit(**fields) -> None:
    """Append one immutable data-access record. Auditing must never break answering."""
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        rec = {"ts": now.isoformat(timespec="seconds"), **fields}
        path = AUDIT_DIR / f"audit-{now.strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass
