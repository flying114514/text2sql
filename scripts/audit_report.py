"""Audit-log report (Phase 10): summarise data-access records.

Governance writes one JSONL record per query to data/audit/audit-YYYYMMDD.jsonl.
This prints them as a compact table so you can answer "who accessed what, and
what did the policy do" — the auditability half of data governance.

    uv run python scripts/audit_report.py            # today
    uv run python scripts/audit_report.py --all       # every day on disk
    uv run python scripts/audit_report.py --date 20260613
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "audit"


def _files(args) -> list[Path]:
    if args.all:
        return sorted(AUDIT_DIR.glob("audit-*.jsonl"))
    day = args.date or datetime.now().strftime("%Y%m%d")
    f = AUDIT_DIR / f"audit-{day}.jsonl"
    return [f] if f.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarise the governance audit log.")
    ap.add_argument("--all", action="store_true", help="include every day on disk")
    ap.add_argument("--date", help="a specific day, format YYYYMMDD")
    args = ap.parse_args()

    files = _files(args)
    if not files:
        print("(no audit records yet — ask a question via the web UI first)")
        return 0

    records = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    blocked = sum(1 for r in records if r.get("blocked"))
    masked = sum(1 for r in records if (r.get("masked_columns")))
    rls = sum(1 for r in records if (r.get("row_security")))
    print(f"共 {len(records)} 条访问记录  ·  拒绝 {blocked}  ·  脱敏 {masked}  ·  行级权限 {rls}\n")

    for r in records:
        flags = []
        if r.get("blocked"):
            flags.append("拒绝")
        if r.get("row_security"):
            flags.append("RLS")
        if r.get("masked_columns"):
            flags.append("脱敏")
        if r.get("dropped_columns"):
            flags.append("删列")
        tag = ("[" + "/".join(flags) + "]") if flags else "[放行]"
        ts = (r.get("ts") or "")[-8:]
        role = r.get("role") or "-"
        q = (r.get("question") or "").replace("\n", " ")
        print(f"{ts}  {tag:<12} {role:<10} {r.get('db_id', '')}/{r.get('row_count', '-')}行  {q}")
        if r.get("blocked") and r.get("reason"):
            print(f"           ↳ {r['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
