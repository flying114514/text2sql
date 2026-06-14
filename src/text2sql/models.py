"""Pydantic data models — the typed contracts that flow through the agent.

Using Pydantic here is an engineering choice, not decoration: the LLM's output
is validated into these structures, so a malformed response fails loudly and
early instead of corrupting downstream steps.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SQLGeneration(BaseModel):
    """The structured result we force the LLM to produce for a question."""

    reasoning: str = Field(
        default="",
        description="Short explanation of how the SQL answers the question.",
    )
    sql: str = Field(description="A single executable SQL query (SQLite dialect).")


class ExecutionResult(BaseModel):
    """The outcome of running a SQL query against a database."""

    ok: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[tuple] = Field(default_factory=list)
    row_count: int = 0
    error: str | None = None
    sql: str = ""

    @property
    def preview(self) -> str:
        """A compact text preview, handy for prompts and logs."""
        if not self.ok:
            return f"ERROR: {self.error}"
        head = self.rows[:10]
        return f"{self.columns}\n" + "\n".join(str(r) for r in head)
