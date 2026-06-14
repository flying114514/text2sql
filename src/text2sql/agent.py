"""The Text2SQL agent.

Phase 1 is deliberately the simplest thing that works end to end:

    question + full schema  ->  LLM (structured JSON)  ->  SQL  ->  execute

No retrieval, no self-correction yet — those are Phase 3 and Phase 4. This
single-shot version is our *baseline*: the number every later improvement is
measured against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .db import execute_sql
from .llm import LLMResponse, complete
from .models import ExecutionResult, SQLGeneration
from .prompts import build_chat_messages, build_messages, build_repair_message
from .schema import format_schema_for_prompt


@dataclass
class AnswerResult:
    """Everything produced for one question — handy for CLI, API and eval."""

    question: str
    generation: SQLGeneration
    execution: ExecutionResult


@dataclass
class CorrectionTrace:
    """A record of the self-correction loop — how many tries, did it recover."""

    attempts: int  # number of LLM calls made (>= 1)
    final_ok: bool  # did the final query execute successfully
    recovered: bool  # the first attempt failed but a later one succeeded
    errors: list[str]  # the error message from each failed attempt, in order


def _strip_code_fences(text: str) -> str:
    """Remove ```...``` fences if a model wrapped its JSON/SQL in them."""
    t = text.strip()
    if t.startswith("```"):
        # drop the first line (``` or ```json) and a trailing ```
        lines = t.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def parse_generation(raw: str) -> SQLGeneration:
    """Parse the model's raw response into a validated SQLGeneration.

    We try strict JSON first (json_mode should give us that). If the provider
    returned something slightly off, we fall back to extracting the outermost
    JSON object before giving up.
    """
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON:\n{raw}") from None
        data = json.loads(cleaned[start : end + 1])

    gen = SQLGeneration.model_validate(data)
    gen.sql = _strip_code_fences(gen.sql).rstrip(";").strip()
    return gen


def generate(
    question: str,
    db_path: str | Path,
    *,
    tables: list[str] | None = None,
    examples: list[dict] | None = None,
) -> tuple[SQLGeneration, LLMResponse]:
    """Generate SQL and also return the raw LLM response (for token usage).

    This is the single place that talks to the model; both the simple
    `generate_sql` helper and the evaluation harness build on it. When `tables`
    is provided (by a Phase 3 retriever), only those tables' DDL is sent —
    otherwise the full schema is dumped (the Phase 1 baseline). When `examples`
    is provided (Phase 4B few-shot), demonstrations are prepended.
    """
    schema = format_schema_for_prompt(db_path, tables=tables)
    messages = build_messages(schema, question, examples=examples)
    resp = complete(messages, json_mode=True)
    return parse_generation(resp.content), resp


def generate_sql(question: str, db_path: str | Path) -> SQLGeneration:
    """Turn a natural-language question into a validated SQL generation."""
    gen, _ = generate(question, db_path)
    return gen


def answer(question: str, db_path: str | Path) -> AnswerResult:
    """End-to-end: generate SQL for the question and execute it."""
    generation = generate_sql(question, db_path)
    execution = execute_sql(db_path, generation.sql)
    return AnswerResult(question=question, generation=generation, execution=execution)


def generate_with_correction(
    question: str,
    db_path: str | Path,
    *,
    tables: list[str] | None = None,
    examples: list[dict] | None = None,
    max_retries: int = 2,
) -> tuple[SQLGeneration | None, LLMResponse, ExecutionResult | None, CorrectionTrace]:
    """The Phase 4 agent loop: generate -> execute -> on error, repair, retry.

    This is what turns the pipeline into an *agent*: it acts (runs SQL), observes
    the result (the DB error), and re-plans (asks the model to fix it), up to
    `max_retries` extra times. The conversation history is kept across attempts
    so the model sees its previous SQL and the exact error it caused.

    Returns the final generation, the *aggregated* token usage across all
    attempts, the final execution result, and a CorrectionTrace for analysis.
    Token aggregation matters: an agent that retries costs more, and the eval
    must account for that honestly.
    """
    schema = format_schema_for_prompt(db_path, tables=tables)
    messages = build_messages(schema, question, examples=examples)

    errors: list[str] = []
    sum_in = sum_out = 0
    last_gen: SQLGeneration | None = None
    last_exec: ExecutionResult | None = None
    last_content = ""

    for attempt in range(max_retries + 1):
        resp = complete(messages, json_mode=True)
        sum_in += resp.prompt_tokens
        sum_out += resp.completion_tokens
        last_content = resp.content

        try:
            gen = parse_generation(resp.content)
        except Exception as e:  # malformed output is also a failure we can repair
            errors.append(f"parse error: {e}")
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(build_repair_message("<unparseable response>", str(e)))
            continue

        last_gen = gen
        exec_result = execute_sql(db_path, gen.sql)
        last_exec = exec_result

        if exec_result.ok:
            agg = LLMResponse(content=last_content, prompt_tokens=sum_in, completion_tokens=sum_out)
            trace = CorrectionTrace(
                attempts=attempt + 1,
                final_ok=True,
                recovered=attempt > 0,
                errors=errors,
            )
            return gen, agg, exec_result, trace

        # Execution failed: hand the error back and let the model try again.
        errors.append(exec_result.error or "unknown error")
        messages.append({"role": "assistant", "content": resp.content})
        messages.append(build_repair_message(gen.sql, exec_result.error or ""))

    # Exhausted all retries without a clean execution.
    agg = LLMResponse(content=last_content, prompt_tokens=sum_in, completion_tokens=sum_out)
    trace = CorrectionTrace(
        attempts=max_retries + 1,
        final_ok=bool(last_exec and last_exec.ok),
        recovered=False,
        errors=errors,
    )
    return last_gen, agg, last_exec, trace


def answer_with_correction(
    question: str,
    db_path: str | Path,
    *,
    tables: list[str] | None = None,
    max_retries: int = 2,
) -> tuple[AnswerResult, CorrectionTrace]:
    """CLI/API-friendly wrapper around the correction loop."""
    gen, _resp, execution, trace = generate_with_correction(
        question, db_path, tables=tables, max_retries=max_retries
    )
    result = AnswerResult(
        question=question,
        generation=gen or SQLGeneration(reasoning="(no valid generation)", sql=""),
        execution=execution or ExecutionResult(ok=False, error="no execution"),
    )
    return result, trace


@dataclass
class ConverseResult:
    """One conversational turn: either SQL, or a request for clarification."""

    kind: str  # "sql" | "clarify"
    sql: str = ""
    reasoning: str = ""
    clarification: str = ""
    response: LLMResponse | None = None


def _loads_json(raw: str) -> dict:
    """Defensive JSON parse shared by the conversational path."""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON:\n{raw}") from None
        return json.loads(cleaned[start : end + 1])


def converse(
    question: str,
    db_path: str | Path,
    *,
    tables: list[str] | None = None,
    examples: list[dict] | None = None,
    semantics: str | None = None,
    history: list[dict] | None = None,
    repair: tuple[str, str] | None = None,
) -> ConverseResult:
    """Conversational generation (P8): resolves follow-ups using `history`, and
    may return a clarifying question instead of SQL when the request is ambiguous.

    P9b: an optional `semantics` block (the business semantic layer) is injected
    so the model uses the right 口径/JOIN/术语 for this database.
    """
    schema = format_schema_for_prompt(db_path, tables=tables)
    messages = build_chat_messages(
        schema,
        question,
        history=history,
        examples=examples,
        semantics=semantics,
        repair=repair,
    )
    resp = complete(messages, json_mode=True)
    data = _loads_json(resp.content)

    if str(data.get("action", "sql")).lower() == "clarify":
        return ConverseResult(
            kind="clarify",
            clarification=str(data.get("clarification", "")).strip(),
            response=resp,
        )

    sql = _strip_code_fences(str(data.get("sql", ""))).rstrip(";").strip()
    return ConverseResult(
        kind="sql", sql=sql, reasoning=str(data.get("reasoning", "")).strip(), response=resp
    )
