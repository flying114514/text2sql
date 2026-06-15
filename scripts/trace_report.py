"""Summarise the local LLM trace logs (Phase 5 observability + P13 gateway).

Reads the JSONL events written by text2sql.tracing and prints an operations-style
report: call volume, success rate, token + cost totals, and latency percentiles.
Grouped both by model and — once the gateway (P13) is in use — by provider, so a
multi-provider deployment can see each backend's reliability and cost side by side.

    uv run python scripts/trace_report.py            # all trace days
    uv run python scripts/trace_report.py 20260613   # one day (YYYYMMDD)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = ROOT / "data" / "traces"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def _group_report(events: list[dict], key_fn, col_label: str) -> None:
    """Print one grouped table (by model or by provider)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        groups[key_fn(e)].append(e)

    header = (
        f"{col_label:22} {'calls':>6} {'ok%':>6} {'tokens':>10} "
        f"{'cost$':>9} {'p50 s':>7} {'p95 s':>7}"
    )
    print(header)
    print("-" * len(header))

    grand_calls = grand_tokens = 0
    grand_cost = 0.0
    for name, evs in sorted(groups.items()):
        calls = len(evs)
        ok = sum(1 for e in evs if e.get("ok"))
        tokens = sum(e.get("total_tokens", 0) for e in evs)
        cost = sum(e.get("cost_usd", 0.0) for e in evs)
        lat = [e.get("latency_s", 0.0) for e in evs]
        print(
            f"{str(name)[:22]:22} {calls:>6} {ok / calls * 100:>5.0f}% {tokens:>10} "
            f"{cost:>9.4f} {percentile(lat, 50):>7.2f} {percentile(lat, 95):>7.2f}"
        )
        grand_calls += calls
        grand_tokens += tokens
        grand_cost += cost

    print("-" * len(header))
    print(f"{'TOTAL':22} {grand_calls:>6} {'':>6} {grand_tokens:>10} {grand_cost:>9.4f}\n")


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else None
    pattern = f"llm-{day}.jsonl" if day else "llm-*.jsonl"
    files = sorted(TRACES_DIR.glob(pattern))
    if not files:
        print(f"No trace files matching {pattern} in {TRACES_DIR}")
        print("Run any agent/eval command first to generate traces.")
        return

    events: list[dict] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Trace report ({len(files)} file(s), {len(events)} calls)\n")

    print("== by model ==")
    _group_report(events, lambda e: e.get("model", "?"), "model")

    # Only show the provider view when the gateway actually tagged calls with one.
    if any(e.get("provider") for e in events):
        print("== by provider (gateway) ==")
        _group_report(events, lambda e: e.get("provider") or "(legacy)", "provider")


if __name__ == "__main__":
    main()
