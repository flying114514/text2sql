"""The evaluation harness — one command, a reproducible report.

    uv run python eval/run_eval.py --dataset mini
    uv run python eval/run_eval.py --dataset spider --limit 100

For every case it: generates SQL (capturing tokens + latency), executes both the
prediction and the gold query, and checks whether the result sets match. It then
prints aggregate metrics and writes a timestamped JSON + JSONL report under
eval/results/ so every run is auditable and comparable over time.

This is the backbone of "data-driven optimization": Phase 3/4 improvements are
judged by re-running this and comparing the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

import datasets as ds  # noqa: E402
from metrics import result_set_match, table_recall  # noqa: E402
from text2sql.agent import generate, generate_with_correction  # noqa: E402
from text2sql.db import execute_sql  # noqa: E402
from text2sql.pricing import estimate_cost  # noqa: E402
from text2sql.retriever import get_retriever  # noqa: E402

console = Console()
RESULTS_DIR = ROOT / "eval" / "results"

# Rough DeepSeek pricing now lives in config (price_in/out_per_m) and is shared
# via text2sql.pricing.estimate_cost so tracing and eval report the same numbers.


def run(
    cases: list[ds.Case],
    label: str,
    retriever=None,
    max_retries: int = 0,
    fewshot=None,
    fewshot_k: int = 0,
) -> dict:
    rows = []
    n = len(cases)
    correct = exec_errors = 0
    sum_latency = sum_in = sum_out = 0
    sum_recall = 0.0
    recall_n = 0  # cases where recall is defined (retrieval on + gold has tables)
    full_recall = 0  # cases where every gold table was retrieved
    sum_attempts = 0  # total LLM attempts (Phase 4: > n means retries happened)
    recovered = 0  # cases that first failed then self-corrected to success

    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        gen_error = None

        # Phase 3: retrieve the relevant tables first (None == full-schema baseline).
        selected = None
        if retriever is not None:
            try:
                selected = retriever.select_tables(case.question, case.db_path)
            except Exception as e:
                gen_error = f"retrieval {type(e).__name__}: {e}"

        # Phase 4B: retrieve few-shot demonstrations (db-disjoint from this case).
        examples = None
        if fewshot is not None and fewshot_k > 0:
            try:
                examples = fewshot.select(case.question, db_id=case.db_id, k=fewshot_k)
            except Exception as e:
                gen_error = gen_error or f"fewshot {type(e).__name__}: {e}"

        pred_exec = None
        attempts = 1
        try:
            if max_retries > 0:
                # Phase 4: generate-execute-repair loop (it executes internally).
                gen, resp, pred_exec, trace = generate_with_correction(
                    case.question,
                    case.db_path,
                    tables=selected,
                    examples=examples,
                    max_retries=max_retries,
                )
                pred_sql = gen.sql if gen else ""
                attempts = trace.attempts
                if trace.recovered:
                    recovered += 1
            else:
                gen, resp = generate(
                    case.question, case.db_path, tables=selected, examples=examples
                )
                pred_sql = gen.sql
            sum_in += resp.prompt_tokens
            sum_out += resp.completion_tokens
        except Exception as e:  # generation/parse failure
            gen_error = gen_error or f"{type(e).__name__}: {e}"
            pred_sql = ""
            resp = None
        latency = time.perf_counter() - t0
        sum_latency += latency
        sum_attempts += attempts

        recall = table_recall(selected, case.gold_sql)
        if recall is not None:
            sum_recall += recall
            recall_n += 1
            if recall >= 1.0:
                full_recall += 1

        # In baseline mode we still need to execute the prediction ourselves.
        if pred_exec is None and pred_sql:
            pred_exec = execute_sql(case.db_path, pred_sql)
        gold_exec = execute_sql(case.db_path, case.gold_sql)

        if pred_exec is not None and not pred_exec.ok:
            exec_errors += 1

        matched = bool(
            pred_exec and result_set_match(pred_exec, gold_exec, order_matters=case.order_matters)
        )
        if matched:
            correct += 1

        rows.append(
            {
                "db_id": case.db_id,
                "question": case.question,
                "gold_sql": case.gold_sql,
                "pred_sql": pred_sql,
                "selected_tables": selected,
                "table_recall": recall,
                "num_examples": len(examples) if examples else 0,
                "attempts": attempts,
                "matched": matched,
                "pred_ok": bool(pred_exec and pred_exec.ok),
                "gold_ok": gold_exec.ok,
                "pred_error": (pred_exec.error if pred_exec else None) or gen_error,
                "latency_s": round(latency, 3),
                "prompt_tokens": resp.prompt_tokens if resp else 0,
                "completion_tokens": resp.completion_tokens if resp else 0,
            }
        )

        status = "PASS" if matched else "FAIL"
        retry_tag = f" [yellow](x{attempts})[/yellow]" if attempts > 1 else ""
        console.print(
            f"[dim]{i:>3}/{n}[/dim] [{'green' if matched else 'red'}]{status}[/]{retry_tag} {case.question[:58]}"
        )

    metrics = {
        "label": label,
        "dataset_size": n,
        "execution_accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "exec_error_rate": round(exec_errors / n, 4) if n else 0.0,
        "avg_latency_s": round(sum_latency / n, 3) if n else 0.0,
        "avg_prompt_tokens": round(sum_in / n, 1) if n else 0.0,
        "avg_completion_tokens": round(sum_out / n, 1) if n else 0.0,
        "total_tokens": sum_in + sum_out,
        "estimated_cost_usd": round(estimate_cost(sum_in, sum_out), 4),
        # Retrieval quality (None when full-schema mode was used).
        "avg_table_recall": round(sum_recall / recall_n, 4) if recall_n else None,
        "full_table_recall_rate": round(full_recall / recall_n, 4) if recall_n else None,
        # Self-correction (None when correction was off).
        "max_retries": max_retries if max_retries > 0 else None,
        "avg_attempts": round(sum_attempts / n, 3) if (n and max_retries > 0) else None,
        "recovered": recovered if max_retries > 0 else None,
        # Few-shot (None when off).
        "fewshot_k": fewshot_k if fewshot_k > 0 else None,
    }
    return {"metrics": metrics, "rows": rows}


def print_summary(metrics: dict) -> None:
    table = Table(title="Evaluation summary", show_header=False)
    table.add_column("metric", style="bold cyan")
    table.add_column("value")
    table.add_row("dataset / label", str(metrics["label"]))
    table.add_row("cases", str(metrics["dataset_size"]))
    table.add_row(
        "execution accuracy",
        f"{metrics['execution_accuracy'] * 100:.1f}%  ({metrics['correct']}/{metrics['dataset_size']})",
    )
    table.add_row("exec error rate", f"{metrics['exec_error_rate'] * 100:.1f}%")
    table.add_row("avg latency", f"{metrics['avg_latency_s']:.2f}s")
    table.add_row(
        "avg tokens (in/out)",
        f"{metrics['avg_prompt_tokens']:.0f} / {metrics['avg_completion_tokens']:.0f}",
    )
    table.add_row("total tokens", str(metrics["total_tokens"]))
    table.add_row("estimated cost (USD)", f"${metrics['estimated_cost_usd']:.4f}")
    if metrics.get("avg_table_recall") is not None:
        table.add_row("avg table recall", f"{metrics['avg_table_recall'] * 100:.1f}%")
        table.add_row("full-recall rate", f"{metrics['full_table_recall_rate'] * 100:.1f}%")
    if metrics.get("max_retries") is not None:
        table.add_row("self-correction", f"on (max_retries={metrics['max_retries']})")
        table.add_row("avg attempts", f"{metrics['avg_attempts']:.3f}")
        table.add_row("recovered by retry", str(metrics["recovered"]))
    if metrics.get("fewshot_k") is not None:
        table.add_row("few-shot examples", f"k={metrics['fewshot_k']} (db-disjoint pool)")
    console.print(table)


def save(report: dict, label: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{label}_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Text2SQL evaluation harness.")
    parser.add_argument("--dataset", choices=["mini", "spider"], default="mini")
    parser.add_argument("--limit", type=int, default=None, help="cap number of cases")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducible sampling")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="take the first N instead of a random sample"
    )
    parser.add_argument(
        "--schema",
        choices=["full", "lexical", "embed"],
        default="full",
        help="schema strategy: full dump (baseline) or a Phase 3 retriever",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="tables a retriever selects before FK expansion"
    )
    parser.add_argument(
        "--correct",
        type=int,
        default=0,
        metavar="N",
        help="Phase 4 self-correction: max repair retries on SQL error (0 = off)",
    )
    parser.add_argument(
        "--fewshot",
        type=int,
        default=0,
        metavar="K",
        help="Phase 4B few-shot: retrieve K db-disjoint example Q->SQL pairs (0 = off)",
    )
    parser.add_argument(
        "--label", default=None, help="label for this run (defaults to dataset+schema)"
    )
    args = parser.parse_args()

    retriever = get_retriever(args.schema, top_k=args.topk)
    fewshot = None
    if args.fewshot > 0:
        from text2sql.examples import get_fewshot_retriever

        fewshot = get_fewshot_retriever(top_k=args.fewshot)

    if args.dataset == "mini":
        cases = ds.load_mini()
    else:
        cases = ds.load_spider()  # load all, then sample below

    # Representative + reproducible sampling: shuffle with a fixed seed, then cap.
    # (Spider's dev.json is grouped by database, so taking the first N would only
    #  cover a couple of databases and easy questions.)
    if args.limit and args.limit < len(cases):
        if not args.no_shuffle:
            random.Random(args.seed).shuffle(cases)
        cases = cases[: args.limit]

    label = args.label or (
        f"{args.dataset}_{args.schema}"
        + (f"_correct{args.correct}" if args.correct else "")
        + (f"_fs{args.fewshot}" if args.fewshot else "")
    )
    console.print(
        f"[bold]Running eval[/bold]: dataset={args.dataset} schema={args.schema} "
        f"correct={args.correct} fewshot={args.fewshot} cases={len(cases)} label={label}\n"
    )
    if args.limit:
        console.print(
            f"[yellow]Note: sampled to {len(cases)} cases (--limit). Not the full set.[/yellow]\n"
        )

    report = run(
        cases,
        label,
        retriever=retriever,
        max_retries=args.correct,
        fewshot=fewshot,
        fewshot_k=args.fewshot,
    )
    console.print()
    print_summary(report["metrics"])
    out = save(report, label)
    console.print(f"\n[dim]report saved: {out}[/dim]")


if __name__ == "__main__":
    main()
