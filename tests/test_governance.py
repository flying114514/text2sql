"""Tests for the Phase 10 data-governance layer.

Enforcement must be deterministic and prompt-independent, so it is exactly the
kind of thing to nail down with unit tests: row-level security rewrites, PII
masking (including alias evasion), column denial, fail-closed parsing, principal
resolution, and audit writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import governance  # noqa: E402
from text2sql.governance import (  # noqa: E402
    Principal,
    _mask_value,
    enforce,
    list_roles,
    load_policies,
    redact,
    resolve_principal,
)
from text2sql.semantics import SemanticLayer  # noqa: E402
from text2sql.sources import DataSource  # noqa: E402

SAMPLE = DataSource(id="sample", label="s", kind="sqlite", dialect="sqlite", path="x.sqlite")

_POLICIES = """
default_role: viewer
identities:
  key-a: analyst
roles:
  analyst:
    label: 分析师
    can_see_pii: true
  viewer:
    label: 访客
    can_see_pii: false
    deny_columns:
      sample:
        - customers.signup_date
  ops:
    label: 运营
    can_see_pii: false
    row_filters:
      sample:
        customers: "city = 'Beijing'"
"""


def _use_policies(tmp_path, monkeypatch, text=_POLICIES):
    f = tmp_path / "policies.yaml"
    f.write_text(text, encoding="utf-8")
    monkeypatch.setattr(governance, "POLICIES_FILE", f)
    load_policies.cache_clear()


def _ops() -> Principal:
    return Principal(
        role="ops", label="运营", can_see_pii=False, row_filters={"customers": "city = 'Beijing'"}
    )


def _viewer_deny() -> Principal:
    return Principal(
        role="viewer", label="访客", can_see_pii=False, deny_columns={"customers.signup_date"}
    )


# --- principal resolution ---------------------------------------------------
def test_resolve_by_role_and_apikey_and_default(tmp_path, monkeypatch):
    _use_policies(tmp_path, monkeypatch)
    assert resolve_principal("analyst", None, "sample").can_see_pii is True
    assert resolve_principal(None, "key-a", "sample").role == "analyst"
    # unknown -> least-privileged default
    p = resolve_principal(None, None, "sample")
    assert p.role == "viewer" and "customers.signup_date" in p.deny_columns


def test_resolve_scopes_rules_to_source(tmp_path, monkeypatch):
    _use_policies(tmp_path, monkeypatch)
    p = resolve_principal("ops", None, "sample")
    assert p.row_filters == {"customers": "city = 'Beijing'"}
    # a source with no rules for this role gets empty filters
    assert resolve_principal("ops", None, "other_db").row_filters == {}


def test_list_roles(tmp_path, monkeypatch):
    _use_policies(tmp_path, monkeypatch)
    roles = {r["role"] for r in list_roles()}
    assert roles == {"analyst", "viewer", "ops"}


def test_no_policy_file_is_permissive(tmp_path, monkeypatch):
    monkeypatch.setattr(governance, "POLICIES_FILE", tmp_path / "nope.yaml")
    load_policies.cache_clear()
    p = resolve_principal("anything", None, "sample")
    assert p.permissive and p.can_see_pii


# --- row-level security (SQL rewrite) ---------------------------------------
def test_rls_wraps_table_in_security_barrier():
    enf = enforce("SELECT id, city FROM customers", _ops(), SAMPLE)
    assert enf.allowed
    assert "Beijing" in enf.sql
    assert enf.row_security == ["customers: city = 'Beijing'"]
    # the rewrite is still valid SQL and still selects from a customers-aliased view
    parsed = sqlglot.parse_one(enf.sql, read="sqlite")
    sub = parsed.find(exp.Subquery)
    assert sub is not None and sub.alias == "customers"


def test_rls_preserves_table_alias():
    enf = enforce("SELECT c.id FROM customers AS c", _ops(), SAMPLE)
    sub = sqlglot.parse_one(enf.sql, read="sqlite").find(exp.Subquery)
    assert sub.alias == "c"  # column refs (c.id) still resolve


def test_rls_only_wraps_governed_table_in_join():
    sql = "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id"
    enf = enforce(sql, _ops(), SAMPLE)
    subs = list(sqlglot.parse_one(enf.sql, read="sqlite").find_all(exp.Subquery))
    assert len(subs) == 1
    assert "customers" in subs[0].sql().lower() and "Beijing" in subs[0].sql()


def test_no_rules_leaves_sql_untouched():
    p = Principal(role="analyst", label="分析师", can_see_pii=True)
    enf = enforce("SELECT * FROM customers", p, SAMPLE)
    assert enf.allowed and enf.sql == "SELECT * FROM customers" and not enf.row_security


# --- column denial ----------------------------------------------------------
def test_denied_column_reference_is_blocked():
    enf = enforce("SELECT name, signup_date FROM customers", _viewer_deny(), SAMPLE)
    assert not enf.allowed
    assert "signup_date" in (enf.reason or "")
    assert enf.blocked_columns == ["signup_date"]


def test_fail_closed_on_unparseable_for_restricted_principal(monkeypatch):
    def _boom(*a, **k):
        raise ValueError("cannot parse")

    monkeypatch.setattr(governance.sqlglot, "parse_one", _boom)
    enf = enforce("@@@ not sql @@@", _ops(), SAMPLE)
    assert not enf.allowed and "拒绝" in (enf.reason or "")


# --- PII masking / redaction ------------------------------------------------
def _patch_pii(monkeypatch, columns=("customers.name",)):
    layer = SemanticLayer(source_id="sample", pii=list(columns))
    monkeypatch.setattr(governance, "load_semantics", lambda _id: layer)


def test_mask_pii_by_column_name(monkeypatch):
    _patch_pii(monkeypatch)
    p = Principal(role="viewer", label="访客", can_see_pii=False)
    cols, rows, masked, dropped = redact(
        ["id", "name"], [(1, "Alice"), (2, "Bob")], p, SAMPLE, "SELECT id, name FROM customers"
    )
    assert masked == ["name"] and dropped == []
    assert rows == [(1, "A****"), (2, "B**")]


def test_mask_is_alias_proof(monkeypatch):
    """`SELECT name AS n` must still be masked — we map projections to sources."""
    _patch_pii(monkeypatch)
    p = Principal(role="viewer", label="访客", can_see_pii=False)
    cols, rows, masked, dropped = redact(
        ["id", "n"], [(1, "Alice")], p, SAMPLE, "SELECT id, name AS n FROM customers"
    )
    assert masked == ["n"] and rows == [(1, "A****")]


def test_can_see_pii_leaves_data_clear(monkeypatch):
    _patch_pii(monkeypatch)
    p = Principal(role="analyst", label="分析师", can_see_pii=True)
    cols, rows, masked, dropped = redact(
        ["id", "name"], [(1, "Alice")], p, SAMPLE, "SELECT id, name FROM customers"
    )
    assert masked == [] and rows == [(1, "Alice")]


def test_denied_column_dropped_from_select_star(monkeypatch):
    _patch_pii(monkeypatch, columns=())  # no PII, isolate the drop
    p = Principal(
        role="viewer", label="访客", can_see_pii=True, deny_columns={"customers.signup_date"}
    )
    cols, rows, masked, dropped = redact(
        ["id", "name", "city", "signup_date"],
        [(1, "A", "BJ", "2020-01-01")],
        p,
        SAMPLE,
        "SELECT * FROM customers",
    )
    assert dropped == ["signup_date"]
    assert cols == ["id", "name", "city"] and rows == [(1, "A", "BJ")]


def test_mask_value_shapes():
    assert _mask_value(None) is None
    assert _mask_value("") == ""
    assert _mask_value("x") == "*"
    assert _mask_value("Bob") == "B**"
    assert _mask_value("Alice") == "A****"
    assert _mask_value("张三") == "张*"


# --- audit ------------------------------------------------------------------
def test_audit_writes_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(governance, "AUDIT_DIR", tmp_path / "audit")
    governance.audit(
        role="viewer", db_id="sample", question="多少客户", ok=True, blocked=False, row_count=5
    )
    files = list((tmp_path / "audit").glob("audit-*.jsonl"))
    assert len(files) == 1
    import json

    rec = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert rec["role"] == "viewer" and rec["row_count"] == 5 and rec["ts"]
