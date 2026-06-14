"""The analyst pass (P7): turn a query result into a plain-language answer.

This is the step that makes the product *data analysis* rather than SQL
generation. After the query executes, we hand the result table back to the model
and ask for a concise, grounded answer plus a few insights — the thing a
non-technical user actually wants. The SQL becomes plumbing they can optionally
inspect.

Grounding matters: the prompt forbids inventing numbers, and we only pass the
real rows. Failure here is non-fatal — the caller falls back to showing the
table without a narrated answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLMResponse, complete
from .prompts import build_answer_messages


@dataclass
class ResultSummary:
    answer: str
    insights: list[str] = field(default_factory=list)
    response: LLMResponse | None = None


def _parse(raw: str) -> tuple[str, list[str]]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return text, []  # last resort: treat the whole thing as the answer
        data = json.loads(text[start : end + 1])
    answer = str(data.get("answer", "")).strip()
    insights = [str(x).strip() for x in (data.get("insights") or []) if str(x).strip()]
    return answer, insights


def summarize_result(
    question: str,
    columns: list[str],
    rows: list,
    *,
    max_rows: int = 50,
) -> ResultSummary:
    """Produce a grounded natural-language answer + insights for a result set."""
    if not columns:
        return ResultSummary(answer="该查询没有返回任何列。")
    if not rows:
        return ResultSummary(answer="没有符合条件的数据。")

    messages = build_answer_messages(question, columns, rows, max_rows=max_rows)
    resp = complete(messages, json_mode=True)
    answer, insights = _parse(resp.content)
    return ResultSummary(answer=answer, insights=insights, response=resp)
