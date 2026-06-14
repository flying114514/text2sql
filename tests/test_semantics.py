"""Tests for the Phase 9b semantic layer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import semantics  # noqa: E402
from text2sql.prompts import build_chat_messages  # noqa: E402
from text2sql.semantics import (  # noqa: E402
    SemanticLayer,
    load_semantics,
    matched_terms,
    render_semantics,
    retrieval_keywords,
)

_YAML = """
description: 电商订单库测试。
glossary:
  - term: GMV
    aliases: [成交额, 销售额]
    means: 已完成订单的销售总额 SUM(order_items.quantity * products.price),status = 'completed'。
metrics:
  - name: 已完成订单
    filter: "orders.status = 'completed'"
joins:
  - "orders.customer_id = customers.id"
rules:
  - 计算销售额默认只算 completed 订单。
pii:
  - customers.name
"""


def _layer_from(tmp_path, monkeypatch, text=_YAML, sid="shop") -> SemanticLayer | None:
    d = tmp_path / "semantics"
    d.mkdir()
    (d / f"{sid}.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(semantics, "SEMANTICS_DIR", d)
    load_semantics.cache_clear()
    return load_semantics(sid)


def test_load_parses_all_sections(tmp_path, monkeypatch):
    layer = _layer_from(tmp_path, monkeypatch)
    assert layer is not None
    assert layer.description.startswith("电商")
    assert layer.glossary[0]["term"] == "GMV"
    assert "orders.customer_id = customers.id" in layer.joins
    assert layer.pii == ["customers.name"]


def test_missing_file_returns_none(tmp_path, monkeypatch):
    d = tmp_path / "semantics"
    d.mkdir()
    monkeypatch.setattr(semantics, "SEMANTICS_DIR", d)
    load_semantics.cache_clear()
    assert load_semantics("nope") is None


def test_matched_terms_by_alias(tmp_path, monkeypatch):
    layer = _layer_from(tmp_path, monkeypatch)
    # alias "销售额" should match the GMV entry
    hits = matched_terms(layer, "这个月的销售额是多少")
    assert len(hits) == 1 and hits[0]["term"] == "GMV"
    assert matched_terms(layer, "有多少客户") == []


def test_retrieval_keywords_surface_real_columns(tmp_path, monkeypatch):
    """A business word ('GMV') must drag in the tables its definition names.

    Keywords are tokenized the same way the retriever tokenizes table docs
    (snake_case split + singularization), so we assert on those word tokens —
    that's exactly what will line up against the tables' column tokens."""
    layer = _layer_from(tmp_path, monkeypatch)
    kw = retrieval_keywords(layer, "本月 GMV 多少")
    for token in ("order", "items", "quantity", "price", "customer"):
        assert token in kw, f"missing {token!r} in {kw!r}"
    # no term matched -> no expansion
    assert retrieval_keywords(layer, "有多少客户") == ""


def test_render_contains_sections(tmp_path, monkeypatch):
    layer = _layer_from(tmp_path, monkeypatch)
    block = render_semantics(layer)
    assert "业务语义层" in block
    assert "GMV" in block and "成交额" in block
    assert "推荐 JOIN" in block
    assert "customers.name" in block


def test_build_chat_messages_injects_semantics():
    block = render_semantics(
        SemanticLayer(
            source_id="x", description="测试库", glossary=[{"term": "GMV", "means": "销售总额"}]
        )
    )
    msgs = build_chat_messages("CREATE TABLE t(a)", "GMV 多少", semantics=block)
    user = msgs[-1]["content"]
    assert "业务语义层" in user and "GMV" in user
    # without semantics, no business block leaks in
    plain = build_chat_messages("CREATE TABLE t(a)", "GMV 多少")
    assert "业务语义层" not in plain[-1]["content"]


def test_real_shop_pg_semantics_loads():
    """The committed semantics/shop_pg.yaml is valid and rich."""
    load_semantics.cache_clear()
    layer = load_semantics("shop_pg")
    assert layer is not None and not layer.is_empty()
    terms = {g["term"] for g in layer.glossary}
    assert "GMV" in terms and "客单价" in terms
