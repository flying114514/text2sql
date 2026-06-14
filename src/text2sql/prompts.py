"""Prompt construction for SQL generation.

Keeping prompts in their own module (rather than inline f-strings scattered in
the agent) is a small but real engineering choice: prompts are the "source
code" of an LLM app, so they deserve one place to read, diff, and version.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert data analyst who writes correct, efficient SQL for SQLite.

You are given a database schema (as CREATE TABLE statements, with a few sample
rows) and a user's question in natural language. Produce ONE SQL query that
answers the question.

Rules:
- Use the SQLite dialect only.
- Use ONLY the exact table and column names that appear in the schema.
- Write a single read-only SELECT query. Never write INSERT/UPDATE/DELETE/DROP.
- Prefer explicit JOINs over implicit ones; alias tables for readability.
- If the question is ambiguous, choose the most reasonable interpretation.
- Do not wrap the SQL in markdown fences.

Respond with a JSON object of EXACTLY this shape:
{
  "reasoning": "<one or two sentences: how the query answers the question>",
  "sql": "<the single SQL query>"
}
"""

USER_TEMPLATE = """\
# Database schema
{schema}

# Question
{question}

Return only the JSON object.
"""


def render_examples(examples: list[dict] | None) -> str:
    """Render retrieved (question -> SQL) demonstrations for the prompt.

    The examples come from OTHER databases, so we only show the question/SQL
    pairs (not their schemas) as style references, and we say so explicitly to
    stop the model from borrowing table names that don't exist here.
    """
    if not examples:
        return ""
    lines = [
        "# Examples from other databases (for SQL style/idioms only — they use",
        "# DIFFERENT schemas, so rely on the schema below for the real question):",
    ]
    for ex in examples:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"SQL: {ex['query']}")
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(
    schema: str,
    question: str,
    examples: list[dict] | None = None,
    semantics: str | None = None,
) -> list[dict]:
    """Assemble the chat messages for one SQL-generation call.

    When `examples` is provided (Phase 4B few-shot retrieval), a demonstrations
    block is prepended to the user turn. When `semantics` is provided (Phase 9b),
    the business semantic layer is placed right before the schema so the model
    reads business定义 → schema → question in that order.
    """
    user_parts = []
    ex_block = render_examples(examples)
    if ex_block:
        user_parts.append(ex_block)
    if semantics:
        user_parts.append(semantics)
    user_parts.append(USER_TEMPLATE.format(schema=schema, question=question))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# --- Self-correction (Phase 4) ---------------------------------------------
# When a generated query fails to execute, we feed the *exact* database error
# back to the model and ask it to fix the query. Showing the failed SQL plus the
# concrete error ("no such column: foo") is what makes the agent loop work: the
# model gets observation feedback, not just "try again".

REPAIR_TEMPLATE = """\
The SQL you produced failed when executed against the database.

Failed SQL:
{sql}

Database error:
{error}

Look carefully at the schema again, find what caused the error (a wrong column
or table name, a bad JOIN, a type mismatch, etc.) and return a corrected query.
Respond with the SAME JSON object shape:
{{
  "reasoning": "<what was wrong and how you fixed it>",
  "sql": "<the corrected SQL query>"
}}
"""


def build_repair_message(sql: str, error: str) -> dict:
    """The user turn that hands a failed attempt's error back to the model."""
    return {"role": "user", "content": REPAIR_TEMPLATE.format(sql=sql, error=error)}


# --- Result analysis (P7: answer, not SQL) ---------------------------------
# After the query runs, a second "analyst" pass turns the raw result table into
# a plain-language answer + a few insights. This is what makes the product a
# data-analysis tool instead of a SQL generator. The model is told to ground
# itself ONLY in the provided rows — it must not invent numbers.

ANSWER_SYSTEM_PROMPT = """\
你是一名数据分析师。给你一个用户的问题,以及为回答它而运行的 SQL 查询返回的结果表。
请用简体中文,基于结果数据,直接回答这个问题。

规则:
- 用数据里的具体数字直接回答问题,口语化、简洁(1~3 句)。
- 绝不能编造结果表里没有的数字或事实。
- 如果结果为空,就说明"没有符合条件的数据"。
- 另外给出 0~3 条简短"洞察"(如最大/最小值、占比、对比、明显趋势),仅在数据确实支持时才给。

只输出如下 JSON 结构:
{
  "answer": "<对问题的直接回答>",
  "insights": ["<洞察1>", "<洞察2>"]
}
"""

ANSWER_USER_TEMPLATE = """\
# 用户问题
{question}

# 查询结果(共 {row_count} 行{truncated_note})
{table}

只输出 JSON 对象。
"""


def _render_result_table(columns: list[str], rows: list, max_rows: int = 50) -> str:
    head = " | ".join(str(c) for c in columns)
    sep = " | ".join("---" for _ in columns)
    body_rows = rows[:max_rows]
    lines = [head, sep]
    for r in body_rows:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    return "\n".join(lines)


def build_answer_messages(
    question: str, columns: list[str], rows: list, max_rows: int = 50
) -> list[dict]:
    truncated_note = f",此处仅展示前 {max_rows} 行" if len(rows) > max_rows else ""
    table = _render_result_table(columns, rows, max_rows=max_rows)
    user = ANSWER_USER_TEMPLATE.format(
        question=question, row_count=len(rows), truncated_note=truncated_note, table=table
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --- Conversational generation (P8: follow-ups + clarification) -------------
# The product path uses this instead of the strict SQL-only prompt: the model
# sees the recent conversation so it can resolve follow-ups ("只看北京"), and it
# may ask ONE clarifying question when the request is genuinely ambiguous.

CHAT_SYSTEM_PROMPT = """\
You are a data-analyst assistant. You turn a user's natural-language question into
ONE read-only SQLite SELECT query against the given schema.

You may also see the recent conversation (the user's previous questions and the
SQL you wrote). Use it to resolve FOLLOW-UP questions: e.g. after "每个城市有多少客户"
a follow-up "只看北京" means add a WHERE filter to the previous query; "按月份分组"
means re-group it. Build on the previous SQL when the new turn clearly continues it.

Choose ONE of two actions:
- "sql": the question is clear enough (possibly via the conversation, or a
  reasonable default assumption) — produce the SQL.
- "clarify": ONLY when the question is genuinely ambiguous AND you cannot make a
  reasonable assumption (an undefined business term, or an unclear choice that
  materially changes the result). Ask ONE short clarifying question, in Chinese.
  Strongly prefer "sql" with a sensible assumption over asking — do not nitpick.

SQL rules: SQLite dialect; only real table/column names from the schema; a single
read-only SELECT; never INSERT/UPDATE/DELETE/DROP; no markdown fences.

Respond with a JSON object of EXACTLY one of these shapes:
{"action": "sql", "reasoning": "<one sentence>", "sql": "<the SQL>"}
{"action": "clarify", "clarification": "<one short question to the user, in Chinese>"}
"""


def _render_history(history: list[dict] | None, max_turns: int = 6) -> str:
    if not history:
        return ""
    items = history[-max_turns:]
    lines = ["# 对话历史(较早在前)"]
    for h in items:
        q = h.get("question", "")
        lines.append(f"用户:{q}")
        if h.get("sql"):
            lines.append(f"助手(已生成 SQL):{h['sql']}")
        elif h.get("clarification"):
            lines.append(f"助手(反问澄清):{h['clarification']}")
    return "\n".join(lines)


def build_chat_messages(
    schema: str,
    question: str,
    *,
    history: list[dict] | None = None,
    examples: list[dict] | None = None,
    semantics: str | None = None,
    repair: tuple[str, str] | None = None,
) -> list[dict]:
    parts: list[str] = []
    ex_block = render_examples(examples)
    if ex_block:
        parts.append(ex_block)
    hist = _render_history(history)
    if hist:
        parts.append(hist)
    if semantics:
        parts.append(semantics)
    parts.append(USER_TEMPLATE.format(schema=schema, question=question))
    if repair:
        sql, error = repair
        parts.append(
            "你上一条 SQL 执行报错了,请修正后重新作答(仍按上面的 JSON 格式)。\n"
            f"出错 SQL:{sql}\n数据库报错:{error}"
        )
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
