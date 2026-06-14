"""Ask the agent a question from the command line.

    uv run python scripts/ask.py "which city has the most customers?"
    uv run python scripts/ask.py            # interactive mode (type questions)

Defaults to the local sample database (data/sample.sqlite). Pass --db to point
at another SQLite file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the console UTF-8 so Chinese / box-drawing characters don't crash on
# Windows' default GBK code page.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from text2sql.agent import (  # noqa: E402
    AnswerResult,
    CorrectionTrace,
    answer,
    answer_with_correction,
)

console = Console()
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "sample.sqlite"


def render(result: AnswerResult, trace: CorrectionTrace | None = None) -> None:
    gen, ex = result.generation, result.execution

    if trace is not None and trace.attempts > 1:
        note = f"took {trace.attempts} attempts" + (
            " — recovered ✅" if trace.recovered else " — still failing"
        )
        console.print(
            Panel(
                "\n".join(f"attempt {i + 1} error: {e}" for i, e in enumerate(trace.errors))
                or note,
                title=f"self-correction ({note})",
                border_style="yellow",
            )
        )

    console.print(Panel(gen.reasoning or "(no reasoning)", title="reasoning", border_style="cyan"))
    console.print(Panel(gen.sql, title="generated SQL", border_style="green"))

    if not ex.ok:
        console.print(Panel(str(ex.error), title="execution error", border_style="red"))
        return

    table = Table(show_header=True, header_style="bold magenta")
    for col in ex.columns:
        table.add_column(str(col))
    for row in ex.rows[:50]:
        table.add_row(*[str(v) for v in row])
    console.print(table)
    console.print(f"[dim]{ex.row_count} row(s)[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Text2SQL agent.")
    parser.add_argument("question", nargs="*", help="natural-language question")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="path to a SQLite database")
    parser.add_argument(
        "--correct",
        type=int,
        default=0,
        metavar="N",
        help="Phase 4 self-correction: max repair retries on SQL error (0 = off)",
    )
    args = parser.parse_args()

    db_path = args.db

    def run_one(q: str) -> None:
        if args.correct > 0:
            result, trace = answer_with_correction(q, db_path, max_retries=args.correct)
            render(result, trace)
        else:
            render(answer(q, db_path))

    if args.question:
        q = " ".join(args.question)
        run_one(q)
        return

    console.print("[bold]Text2SQL agent[/bold] — type a question (empty line to quit)\n")
    while True:
        try:
            q = console.input("[bold yellow]?[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            run_one(q)
        except Exception as e:  # noqa: BLE001
            console.print(Panel(f"{type(e).__name__}: {e}", title="error", border_style="red"))
        console.print()


if __name__ == "__main__":
    main()
