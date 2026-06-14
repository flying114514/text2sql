"""Semantic layer (Phase 9b): the business knowledge a schema cannot express.

A schema tells the model the *shape* of the data — tables, columns, types. It
does NOT tell it that "GMV" means SUM(quantity * price) over completed orders,
that "活跃客户" has a specific definition, or which JOIN path is the intended
one. Without that, a model writes SQL that is syntactically valid but answers the
wrong question — it gets the **口径 (definition)** wrong, which is the #1 reason
Text2SQL fails in real businesses.

This module loads a per-source YAML that captures that knowledge:
  * glossary — business terms + aliases + how to express them in THIS database
  * metrics  — reusable metric definitions (a SQL fragment or a filter clause)
  * joins    — recommended join paths, so the model does not guess relationships
  * rules    — default conventions (e.g. "GMV counts only completed orders")
  * pii      — sensitive columns (declared here; actual masking arrives in P10)

The rendered block is injected into the generation prompt. Term hits also expand
the retrieval query, so a question about "GMV" can still pull in the order/
product tables even though that word appears in no column name.

Files live in `semantics/<source_id>.yaml`. Unlike connections.yaml these are
**business definitions, not secrets**, so they are committed to the repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .config import ROOT_DIR
from .schema_index import tokenize

SEMANTICS_DIR = ROOT_DIR / "semantics"


@dataclass
class SemanticLayer:
    """Parsed business semantics for one data source. Every section optional."""

    source_id: str
    description: str = ""
    glossary: list[dict] = field(default_factory=list)  # {term, aliases[], means}
    metrics: list[dict] = field(default_factory=list)  # {name, sql?, filter?, note?}
    joins: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.description
            or self.glossary
            or self.metrics
            or self.joins
            or self.rules
            or self.pii
        )


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


@lru_cache(maxsize=64)
def load_semantics(source_id: str) -> SemanticLayer | None:
    """Load semantics/<id>.yaml for a source, or None if there is no file."""
    path = SEMANTICS_DIR / f"{source_id}.yaml"
    if not path.exists():
        return None
    import yaml  # local import: only needed when a semantics file exists

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 — a broken file must not crash answering
        raise RuntimeError(f"failed to parse {path.name}: {e}") from e

    layer = SemanticLayer(
        source_id=source_id,
        description=str(data.get("description") or "").strip(),
        glossary=[g for g in _as_list(data.get("glossary")) if isinstance(g, dict)],
        metrics=[m for m in _as_list(data.get("metrics")) if isinstance(m, dict)],
        joins=[str(j).strip() for j in _as_list(data.get("joins")) if str(j).strip()],
        rules=[str(r).strip() for r in _as_list(data.get("rules")) if str(r).strip()],
        pii=[str(p).strip() for p in _as_list(data.get("pii")) if str(p).strip()],
    )
    return None if layer.is_empty() else layer


def _term_names(entry: dict) -> list[str]:
    """All the surface forms of a glossary term: its name plus aliases."""
    names = [str(entry.get("term") or "").strip()]
    names += [str(a).strip() for a in _as_list(entry.get("aliases"))]
    return [n for n in names if n]


def matched_terms(layer: SemanticLayer | None, question: str) -> list[dict]:
    """Glossary entries whose term or any alias appears in the question."""
    if not layer:
        return []
    q = question.lower()
    hits = []
    for entry in layer.glossary:
        if any(name.lower() in q for name in _term_names(entry)):
            hits.append(entry)
    return hits


def retrieval_keywords(layer: SemanticLayer | None, question: str) -> str:
    """Extra text to append to the retrieval query for matched business terms.

    A term like "GMV" shares no token with any column, so lexical retrieval would
    miss the order/product tables. Appending the term's definition (which names
    real tables/columns) lets the existing retriever surface them — the semantic
    layer improving retrieval, not just generation."""
    extra: list[str] = []
    for entry in matched_terms(layer, question):
        extra.append(str(entry.get("means") or ""))
    # also fold in join targets so bridge tables can be recalled
    if extra and layer:
        extra.extend(layer.joins)
    text = " ".join(extra).strip()
    # keep only tokens (drops punctuation/operators) so it blends with the query
    return " ".join(sorted(tokenize(text))) if text else ""


def render_semantics(layer: SemanticLayer | None) -> str:
    """Render the semantic layer as a prompt block (Chinese, business-facing)."""
    if not layer or layer.is_empty():
        return ""
    lines = ["# 业务语义层(本库专属知识 —— 理解问题口径时优先参考)"]
    if layer.description:
        lines.append(f"库说明:{layer.description}")

    if layer.glossary:
        lines.append("\n## 业务术语(用户口中的词 → 在本库如何表达)")
        for g in layer.glossary:
            term = str(g.get("term") or "").strip()
            aliases = [str(a).strip() for a in _as_list(g.get("aliases")) if str(a).strip()]
            means = str(g.get("means") or "").strip()
            alias_note = f"(别名:{'/'.join(aliases)})" if aliases else ""
            lines.append(f"- {term}{alias_note}:{means}")

    if layer.metrics:
        lines.append("\n## 指标口径")
        for m in layer.metrics:
            name = str(m.get("name") or "").strip()
            parts = []
            if m.get("sql"):
                parts.append(f"计算式 `{str(m['sql']).strip()}`")
            if m.get("filter"):
                parts.append(f"筛选条件 `{str(m['filter']).strip()}`")
            if m.get("note"):
                parts.append(str(m["note"]).strip())
            lines.append(f"- {name}:" + ";".join(parts))

    if layer.joins:
        lines.append("\n## 推荐 JOIN 路径(按此连接,勿臆测关系)")
        for j in layer.joins:
            lines.append(f"- {j}")

    if layer.rules:
        lines.append("\n## 口径规则(务必遵守)")
        for r in layer.rules:
            lines.append(f"- {r}")

    if layer.pii:
        lines.append("\n## 敏感字段(PII)")
        lines.append("无明确需要时,不要在 SELECT 中返回这些列的明文:" + "、".join(layer.pii))

    return "\n".join(lines).strip()
