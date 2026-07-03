"""Service layer (Phase 6): one call that runs the whole agent for the UI/API.

This wires together everything built so far — schema retrieval (Phase 3),
few-shot (Phase 4B), self-correction (Phase 4), read-only execution + guard +
timeout (Phase 5) — behind a single `answer_query()` that returns a JSON-friendly
dict, plus a small chart suggestion so the frontend can visualise results.
"""

from __future__ import annotations

import re
import time

from . import cache, governance
from .agent import converse  # noqa: F401
from .db import execute_sql
from .pricing import estimate_cost
from .retriever import get_retriever
from .sources import DataSource, get_source, list_sources

try:
    from langfuse import observe
except ImportError:
    def observe(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# --- database discovery -----------------------------------------------------
def list_databases() -> list[dict]:
    """Find the databases the UI can query (local SQLite + external sources)."""
    return [
        {"id": s.id, "label": s.label, "kind": s.kind, "dialect": s.dialect} for s in list_sources()
    ]


def _resolve_db(db_id: str) -> DataSource | None:
    return get_source(db_id)


def _governance_blocked(
    question, db_id, sql, principal, enf, t0, in_tokens, out_tokens, turn
) -> dict:
    """Build the response for a query refused by row/column policy (P10)."""
    return {
        "ok": False,
        "action": "sql",
        "governance_blocked": True,
        "question": question,
        "db_id": db_id,
        "sql": sql,
        "error": enf.reason,
        "role": principal.role,
        "role_label": principal.label,
        "governance": {
            "row_security": [],
            "masked_columns": [],
            "dropped_columns": enf.blocked_columns,
        },
        "model": turn.response.model if turn.response else "",
        "latency_s": round(time.perf_counter() - t0, 3),
        "tokens": {"in": in_tokens, "out": out_tokens, "total": in_tokens + out_tokens},
        "cost_usd": round(estimate_cost(in_tokens, out_tokens), 6),
    }


# --- chart suggestion -------------------------------------------------------
def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_id_like(name: str) -> bool:
    """Columns that are identifiers, not measures — bad y-axis choices."""
    n = name.strip().lower()
    return n in ("id", "key") or n.endswith("_id") or n.endswith("_key")


# Words that mark a column as the metric the user actually asked about.
_MEASURE_HINTS = (
    "count",
    "cnt",
    "total",
    "sum",
    "amount",
    "avg",
    "average",
    "mean",
    "num",
    "number",
    "qty",
    "quantity",
    "price",
    "revenue",
    "sales",
    "value",
    "score",
    "rate",
    "ratio",
    "min",
    "max",
)


def _measure_score(name: str, idx: int) -> float:
    """Rank a numeric column's fitness as the bar height. Id-like columns sink;
    columns whose name reads like a metric float to the top; ties broken by
    position (aggregates tend to be the last column)."""
    n = name.strip().lower()
    if _is_id_like(n):
        return -100.0 + idx
    score = 0.0
    if any(h in n for h in _MEASURE_HINTS):
        score += 10.0
    return score + idx * 0.1


def _has_order_by(sql: str | None) -> bool:
    return bool(sql and re.search(r"\border\s+by\b", sql, re.IGNORECASE))


def suggest_chart(columns: list[str], rows: list[tuple], sql: str | None = None) -> dict | None:
    """Heuristic: if there's a label column + a numeric column over a handful of
    rows, suggest a bar chart. Otherwise no chart (e.g. a single aggregate).

    The y-axis is the *measure* the user asked about (id-like columns are
    skipped). Bars are sorted descending by value so the chart reads as a
    ranking — unless the SQL already specified ORDER BY, in which case that
    intentional order (e.g. a top-N ranking or a time series) is preserved.
    """
    if not columns or not (2 <= len(rows) <= 50):
        return None

    n = len(columns)
    # Classify each column by sampling its values.
    numeric_cols, label_cols = [], []
    for j in range(n):
        vals = [r[j] for r in rows if j < len(r) and r[j] is not None]
        if vals and all(_is_number(v) for v in vals):
            numeric_cols.append(j)
        elif vals and all(isinstance(v, str) for v in vals):
            label_cols.append(j)

    if not numeric_cols or not label_cols:
        return None

    x_idx = label_cols[0]
    # Pick the best measure column for the bar heights (not an id).
    y_idx = max(
        (j for j in numeric_cols if j != x_idx),
        key=lambda j: _measure_score(columns[j], j),
        default=None,
    )
    if y_idx is None:
        return None

    pairs = [(str(r[x_idx]), r[y_idx] if _is_number(r[y_idx]) else 0) for r in rows]
    if not _has_order_by(sql):
        pairs.sort(key=lambda p: p[1], reverse=True)

    return {
        "type": "bar",
        "x_label": columns[x_idx],
        "y_label": columns[y_idx],
        "x": [p[0] for p in pairs],
        "y": [p[1] for p in pairs],
    }


def _json_safe_rows(rows: list[tuple], limit: int = 200) -> list[list]:
    safe: list[list] = []
    for r in rows[:limit]:
        safe.append([v.hex() if isinstance(v, (bytes, bytearray)) else v for v in r])
    return safe


# --- main entry point -------------------------------------------------------
@observe()
def answer_query(
    question: str,
    db_id: str,
    *,
    schema: str = "lexical",
    fewshot: int = 0,
    correct: int = 0,
    analyze: bool = True,
    history: list[dict] | None = None,
    role: str | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    provider_id: str | None = None,
) -> dict:
    """Run the full pipeline for one question and return a UI-ready result.

    P8: `history` (prior {question, sql|clarification} turns) lets the agent
    resolve follow-ups, and the agent may return a clarification instead of SQL.

    P10: `role`/`api_key` resolve a governance Principal. Column denial + row-
    level security are enforced on the SQL before execution; PII is masked and
    denied columns dropped from the result; every access is audited.

    P11: an answer cache short-circuits a repeated *first-turn* question for the
    *same role* (see cache.py for why both qualifiers matter).
    """
    source = _resolve_db(db_id)
    if source is None:
        return {"ok": False, "error": f"unknown database: {db_id!r}"}

    principal = governance.resolve_principal(role, api_key, source.id)

    t0 = time.perf_counter()
    in_tokens = out_tokens = 0

    # P11: cache key components that change the answer (role lives inside the
    # key via principal). Follow-ups depend on history, so we never cache them.
    cache_role = None if principal.permissive else principal.role
    knobs = {"schema": schema, "fewshot": fewshot, "correct": correct, "analyze": analyze}
    cacheable = use_cache and not history

    if cacheable:
        hit = cache.get(question, source.id, cache_role, knobs)
        if hit is not None:
            payload = dict(hit)
            payload["cached"] = True
            payload["latency_s"] = round(time.perf_counter() - t0, 3)
            payload["cost_usd"] = 0.0
            payload["tokens"] = {"in": 0, "out": 0, "total": 0}
            # A cache hit is still a data access — audit it (with the cached flag).
            governance.audit(
                role=cache_role,
                db_id=db_id,
                question=question,
                sql_model=payload.get("sql"),
                sql_executed=payload.get("executed_sql") or payload.get("sql"),
                ok=payload.get("ok"),
                blocked=False,
                cached=True,
                row_security=(payload.get("governance") or {}).get("row_security"),
                masked_columns=(payload.get("governance") or {}).get("masked_columns"),
                dropped_columns=(payload.get("governance") or {}).get("dropped_columns"),
                row_count=payload.get("row_count"),
            )
            return payload

    def _acc(resp):
        nonlocal in_tokens, out_tokens
        if resp:
            in_tokens += resp.prompt_tokens
            out_tokens += resp.completion_tokens

    retriever = get_retriever(schema)
    tables = None

    # P9b: load this source's business semantic layer (if any).
    from .semantics import load_semantics, render_semantics, retrieval_keywords

    layer = load_semantics(source.id)
    semantics_block = render_semantics(layer) if layer else None

    if retriever is not None:
        try:
            # Term hits expand the retrieval query so business words ("GMV")
            # can still surface the tables their definitions reference.
            kw = retrieval_keywords(layer, question)
            q_for_retrieval = f"{question} {kw}".strip() if kw else question
            tables = retriever.select_tables(q_for_retrieval, source)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"retrieval failed: {e}"}

    # P11b: verified same-db examples from the 👍 flywheel come first (highest
    # trust — human-approved on this very schema). The generic Spider pool
    # (cross-db, leakage-guarded) fills the rest. Either source may be empty.
    from .feedback import learned_examples

    learned = learned_examples(source.id, question)
    spider: list[dict] = []
    if fewshot > 0:
        try:
            from .examples import get_fewshot_retriever

            spider = (
                get_fewshot_retriever(top_k=fewshot).select(question, db_id=db_id, k=fewshot) or []
            )
        except Exception:
            spider = []  # pool not downloaded -> just skip Spider few-shot
    examples = (learned + spider) or None

    # P8: conversational generation — may ask for clarification.
    try:
        turn = converse(
            question,
            source,
            tables=tables,
            examples=examples,
            semantics=semantics_block,
            history=history,
            principal=principal,
            provider_id=provider_id,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"generation failed: {e}",
            "latency_s": round(time.perf_counter() - t0, 3),
        }
    _acc(turn.response)

    if turn.kind == "clarify":
        return {
            "ok": True,
            "action": "clarify",
            "clarification": turn.clarification,
            "question": question,
            "db_id": db_id,
            "model": turn.response.model if turn.response else "",
            "latency_s": round(time.perf_counter() - t0, 3),
            "tokens": {"in": in_tokens, "out": out_tokens, "total": in_tokens + out_tokens},
            "cost_usd": round(estimate_cost(in_tokens, out_tokens), 6),
        }

    def _finish(payload: dict) -> dict:
        """Stamp shared fields, write the audit record, and return."""
        governance.audit(
            role=principal.role if not principal.permissive else None,
            db_id=db_id,
            question=question,
            sql_model=payload.get("sql"),
            sql_executed=payload.get("executed_sql") or payload.get("sql"),
            ok=payload.get("ok"),
            blocked=payload.get("governance_blocked", False),
            reason=payload.get("error") if payload.get("governance_blocked") else None,
            row_security=(payload.get("governance") or {}).get("row_security"),
            masked_columns=(payload.get("governance") or {}).get("masked_columns"),
            dropped_columns=(payload.get("governance") or {}).get("dropped_columns"),
            row_count=payload.get("row_count"),
        )
        return payload

    # Execute, with optional self-correction retries (still conversational).
    # Governance (P10) is enforced on every attempt's SQL, before it runs.
    sql = turn.sql
    reasoning = turn.reasoning
    enf = governance.enforce(sql, principal, source)
    if not enf.allowed:
        return _finish(
            _governance_blocked(
                question, db_id, sql, principal, enf, t0, in_tokens, out_tokens, turn
            )
        )
    executed_sql = enf.sql
    row_security = enf.row_security
    execu = execute_sql(source, executed_sql)
    attempts = 1
    retries = correct
    while not execu.ok and retries > 0:
        attempts += 1
        turn = converse(
            question,
            source,
            tables=tables,
            examples=examples,
            semantics=semantics_block,
            history=history,
            repair=(sql, execu.error or ""),
            principal=principal,
            provider_id=provider_id,
        )
        _acc(turn.response)
        if turn.kind != "sql" or not turn.sql:
            break
        sql, reasoning = turn.sql, turn.reasoning
        enf = governance.enforce(sql, principal, source)
        if not enf.allowed:
            return _finish(
                _governance_blocked(
                    question, db_id, sql, principal, enf, t0, in_tokens, out_tokens, turn
                )
            )
        executed_sql, row_security = enf.sql, enf.row_security
        execu = execute_sql(source, executed_sql)
        retries -= 1

    columns = execu.columns if execu else []
    rows = execu.rows if execu else []
    ok = bool(execu and execu.ok)

    # P10: mask PII / drop denied columns *before* anything downstream sees the
    # data — so the analyst pass can never leak a name into the answer text.
    masked_cols: list[str] = []
    dropped_cols: list[str] = []
    if ok:
        columns, rows, masked_cols, dropped_cols = governance.redact(
            columns, rows, principal, source, executed_sql
        )

    # P7: the analyst pass — turn the result table into a plain-language answer.
    answer_text, insights = None, []
    if analyze and ok:
        try:
            from .analyst import summarize_result

            summary = summarize_result(question, columns, rows)
            answer_text, insights = summary.answer, summary.insights
            _acc(summary.response)
        except Exception:
            answer_text, insights = None, []  # non-fatal: fall back to the table

    latency = time.perf_counter() - t0

    payload = _finish(
        {
            "ok": ok,
            "action": "sql",
            "question": question,
            "db_id": db_id,
            "answer": answer_text,
            "insights": insights,
            "reasoning": reasoning,
            "sql": sql,
            "executed_sql": executed_sql if executed_sql != sql else None,
            "columns": columns,
            "rows": _json_safe_rows(rows),
            "row_count": execu.row_count if execu else 0,
            "error": (execu.error if execu and not execu.ok else None),
            "tables_used": tables,
            "num_examples": len(examples) if examples else 0,
            "num_learned": len(learned),
            "semantics_used": bool(semantics_block),
            "attempts": attempts,
            "cached": False,
            "role": None if principal.permissive else principal.role,
            "role_label": None if principal.permissive else principal.label,
            "governance": None
            if principal.permissive
            else {
                "row_security": row_security,
                "masked_columns": masked_cols,
                "dropped_columns": dropped_cols,
            },
            "model": turn.response.model if turn.response else "",
            "latency_s": round(latency, 3),
            "tokens": {"in": in_tokens, "out": out_tokens, "total": in_tokens + out_tokens},
            "cost_usd": round(estimate_cost(in_tokens, out_tokens), 6),
            "chart": suggest_chart(columns, rows, sql=sql),
        }
    )

    # P11: cache only successful, non-blocked, first-turn answers. The stored
    # payload is already governed for this role, so a later hit is safe to serve.
    if cacheable and payload.get("ok"):
        cache.put(question, source.id, cache_role, knobs, payload)
    return payload
