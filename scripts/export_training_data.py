"""导出反馈飞轮数据为 Alpaca 格式，供 LLaMA-Factory QLoRA 微调使用。

    uv run python scripts/export_training_data.py
    uv run python scripts/export_training_data.py --min-samples 50 --output data/finetune/train.jsonl

输出格式（Alpaca）:
    {"instruction": "将中文问题转换为 SQL 查询", "input": "每个城市有多少客户？", "output": "SELECT city, COUNT(*) AS cnt FROM customers GROUP BY city ORDER BY cnt DESC"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# LLaMA-Factory 默认 system prompt（保持与 Qwen2.5 训练一致）
SYSTEM_PROMPT = (
    "You are a Text-to-SQL assistant. Given a database schema and a natural language "
    "question, generate a single, executable SQL query that answers the question correctly."
)

OUTPUT_DIR = ROOT / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_learned() -> dict[str, dict[str, dict]]:
    learned_path = ROOT / "data" / "feedback" / "learned.json"
    if not learned_path.exists():
        print(f"[!] learned.json not found at {learned_path}")
        return {}
    return json.loads(learned_path.read_text(encoding="utf-8"))


def export_alpaca(min_samples: int = 1, output_path: Path | None = None) -> int:
    """导出所有数据库的 👍 数据为 Alpaca 格式。返回导出条数。"""
    learned = load_learned()
    records: list[dict] = []

    for db_id, bucket in learned.items():
        if db_id == "sample" and not bucket:
            continue  # skip placeholder
        for norm_q, item in bucket.items():
            question = item.get("question", "").strip()
            sql = item.get("query", "").strip()
            if not question or not sql:
                continue
            records.append({
                "instruction": SYSTEM_PROMPT,
                "input": question,
                "output": sql,
                # 保留元数据用于审计追踪
                "metadata": {
                    "db_id": db_id,
                    "verified_at": item.get("ts"),
                    "source": "feedback_flywheel",
                },
            })

    if len(records) < min_samples:
        print(f"[!] Only {len(records)} samples (min required: {min_samples}), skipping export")
        return 0

    output = output_path or (OUTPUT_DIR / "train.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[[OK]] Exported {len(records)} Alpaca records to {output}")
    print(f"    Source: {len(learned)} database(s)")
    for db_id, bucket in sorted(learned.items()):
        if bucket:
            print(f"      {db_id}: {len(bucket)} examples")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Export feedback data for fine-tuning")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="Minimum samples required to export (default: 1)")
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_DIR / "train.jsonl",
                        help="Output path for Alpaca JSONL")
    parser.add_argument("--stats", action="store_true",
                        help="Only show stats, don't export")
    args = parser.parse_args()

    learned = load_learned()
    total = sum(len(b) for b in learned.values() if isinstance(b, dict))
    by_db = {db: len(b) for db, b in learned.items() if isinstance(b, dict) and b}

    if args.stats:
        print(f"Feedback flywheel stats:")
        print(f"  Total verified examples: {total}")
        for db_id, count in sorted(by_db.items()):
            print(f"    {db_id}: {count}")
        return

    if total == 0:
        print("[!] No verified examples found. Collect some 👍 feedback first.")
        return

    export_alpaca(min_samples=args.min_samples, output_path=args.output)


if __name__ == "__main__":
    main()
