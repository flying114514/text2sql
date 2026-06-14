"""Dataset loaders for evaluation.

The harness is intentionally decoupled from any specific dataset: everything is
a `Case`. Today we support two sources:
  * the local "mini" set (hand-written gold SQL over our sample.sqlite),
  * the Spider benchmark dev split.
Adding a new dataset = writing one more loader that yields Cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass
class Case:
    db_id: str
    db_path: Path
    question: str
    gold_sql: str
    order_matters: bool = False


def load_mini() -> list[Case]:
    """Load the local mini eval set defined in eval/mini_eval.json."""
    path = Path(__file__).resolve().parent / "mini_eval.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[Case] = []
    for item in raw:
        cases.append(
            Case(
                db_id=item["db_id"],
                db_path=DATA / item["db_path"],
                question=item["question"],
                gold_sql=item["gold_sql"],
                order_matters=item.get("order_matters", False),
            )
        )
    return cases


def load_spider(limit: int | None = None, spider_dir: Path | None = None) -> list[Case]:
    """Load the Spider dev split.

    Expects the standard Spider layout under data/spider/:
        data/spider/dev.json
        data/spider/database/<db_id>/<db_id>.sqlite
    `limit` caps the number of cases (cost/time control) — and we always log
    when we sample, never silently truncate.
    """
    spider_dir = spider_dir or (DATA / "spider")
    dev_path = spider_dir / "dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(
            f"Spider dev.json not found at {dev_path}.\n"
            "Download Spider into data/spider/ first (see eval/README or 项目设计书)."
        )

    rows = json.loads(dev_path.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]

    cases: list[Case] = []
    for item in rows:
        db_id = item["db_id"]
        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        cases.append(
            Case(
                db_id=db_id,
                db_path=db_path,
                question=item["question"],
                gold_sql=item["query"],
                order_matters="order by" in item["query"].lower(),
            )
        )
    return cases
