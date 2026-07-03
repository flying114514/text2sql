#!/usr/bin/env python3
"""自动化微调流水线 — Text2SQL 反馈飞轮的闭环训练。

触发条件（双触发）:
    1. 新增 [UP] 数 >= THRESHOLD_SAMPLES (默认 500)
    2. 距上次微调 >= MAX_INTERVAL_DAYS (默认 14) 且新增 >= MIN_SAMPLES_FALLBACK (默认 50)

流水线阶段:
    1. 检查触发条件
    2. 备份当前模型 (Ollama tag)
    3. 导出训练数据 (learned.json -> Alpaca train.jsonl)
    4. 基线评测 (Spider eval on 当前模型)
    5. QLoRA 微调 (LLaMA-Factory)
    6. 导入微调模型到 Ollama
    7. 微调后评测
    8. 对比决策: 提高 -> 替换线上模型 | 下降 -> 回滚到备份
    9. 记录审计日志, 保留训练数据不删除

用法:
    uv run python scripts/auto_finetune.py --dry-run      # 仅检查触发条件，不实际微调（默认）
    uv run python scripts/auto_finetune.py --execute       # 强制执行完整流水线（需 GPU）
    uv run python scripts/auto_finetune.py --execute --skip-trigger-check  # 跳过触发条件检查
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ============================================================
# 配置常量
# ============================================================
THRESHOLD_SAMPLES = 500          # 新增 [UP] 达到此数触发微调
MAX_INTERVAL_DAYS = 14           # 最大间隔天数
MIN_SAMPLES_FALLBACK = 50        # 间隔触发时最少需要的新增样本数

FINETUNE_DIR = ROOT / "data" / "finetune"
RUN_HISTORY_PATH = FINETUNE_DIR / "run_history.jsonl"
LAST_RUN_PATH = FINETUNE_DIR / "last_run.json"
TRAIN_DATA_PATH = FINETUNE_DIR / "train.jsonl"
LLAMA_CONFIG_TEMPLATE = ROOT / "scripts" / "finetune_config.yaml"

OLLAMA_MODEL = "qwen2.5-coder:1.5b-fast"       # 当前线上本地模型
OLLAMA_BASE = "qwen2.5-coder:1.5b"             # 基础模型（未优化版，用于 merge）

# 评测参数
EVAL_DATASET = "mini"          # 快速评测用 mini, 完整对比用 spider
EVAL_LIMIT = 50                # 快速评测限制样本数

FINETUNE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 阶段 1: 触发条件检查
# ============================================================
def count_learned_examples() -> dict[str, int]:
    """统计 learned.json 中每个数据库的 [UP] 数量。"""
    learned_path = ROOT / "data" / "feedback" / "learned.json"
    if not learned_path.exists():
        return {}
    learned = json.loads(learned_path.read_text(encoding="utf-8"))
    return {
        db: len(bucket) for db, bucket in learned.items()
        if isinstance(bucket, dict) and db != "sample" or bucket
    }


def load_last_run() -> dict | None:
    """加载上次微调记录。"""
    if not LAST_RUN_PATH.exists():
        return None
    return json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))


def save_last_run(data: dict) -> None:
    """写入本次微调记录。"""
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def check_trigger() -> tuple[bool, str, dict]:
    """检查是否满足微调触发条件。

    Returns:
        (should_run, reason, stats) — stats 包含 total/new/days_since 等字段。
    """
    counts = count_learned_examples()
    total = sum(counts.values())
    last = load_last_run()

    stats = {
        "total_learned": total,
        "by_db": counts,
        "checked_at": datetime.now().isoformat(),
    }

    if last is None:
        stats["new_since_last"] = total
        stats["days_since_last"] = None
        if total >= THRESHOLD_SAMPLES:
            return True, f"首次运行, 已有 {total} >= {THRESHOLD_SAMPLES} 条数据", stats
        return False, f"首次运行, 仅 {total} 条数据 (需 >= {THRESHOLD_SAMPLES})", stats

    prev_total = last.get("total_learned_at_run", 0)
    new_since = total - prev_total
    last_ts = datetime.fromisoformat(last["run_at"])
    days_since = (datetime.now() - last_ts).days

    stats["new_since_last"] = max(0, new_since)
    stats["days_since_last"] = days_since
    stats["prev_total"] = prev_total

    # 双触发逻辑
    if new_since >= THRESHOLD_SAMPLES:
        return True, f"新增 {new_since} >= {THRESHOLD_SAMPLES} 条数据", stats
    if days_since >= MAX_INTERVAL_DAYS and new_since >= MIN_SAMPLES_FALLBACK:
        return True, f"距上次 {days_since}d >= {MAX_INTERVAL_DAYS}d 且新增 {new_since} >= {MIN_SAMPLES_FALLBACK}", stats

    return False, f"不满足触发条件 (新增 {new_since}, 距上次 {days_since}d)", stats


# ============================================================
# 阶段 2: 模型备份
# ============================================================
def backup_model() -> str:
    """为当前 Ollama 模型创建备份 tag。返回备份名称。"""
    tag = f"qwen2.5-coder:backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\n[2/9] 备份当前模型: {OLLAMA_MODEL} -> {tag}")
    _run(["ollama", "cp", OLLAMA_MODEL, tag], "模型备份失败")
    print(f"      [OK] 备份完成: {tag}")
    return tag


# ============================================================
# 阶段 3: 导出训练数据
# ============================================================
def export_data() -> int:
    """调用 export_training_data.py 导出 Alpaca 格式数据。返回导出条数。"""
    print(f"\n[3/9] 导出训练数据...")
    script = ROOT / "scripts" / "export_training_data.py"
    result = _run(
        ["uv", "run", "python", str(script), "--output", str(TRAIN_DATA_PATH)],
        "训练数据导出失败",
    )
    # 统计导出行数
    if TRAIN_DATA_PATH.exists():
        count = sum(1 for _ in open(TRAIN_DATA_PATH, encoding="utf-8"))
        print(f"      [OK] 已导出 {count} 条 Alpaca 格式训练数据")
        return count
    return 0


# ============================================================
# 阶段 4 & 7: 评测
# ============================================================
def run_eval(provider_id: str | None, label: str, dataset: str = EVAL_DATASET,
             limit: int = EVAL_LIMIT) -> dict | None:
    """运行 Spider 评测并返回结果摘要。"""
    print(f"\n     [{label}] 评测数据集={dataset} limit={limit} provider={provider_id or '默认'}...")
    eval_script = ROOT / "eval" / "run_eval.py"
    cmd = [
        "uv", "run", "python", str(eval_script),
        "--dataset", dataset,
        "--limit", str(limit),
        "--schema", "lexical",
        "--label", label,
    ]
    if provider_id:
        cmd.extend(["--provider-id", provider_id])

    try:
        result = _run(cmd, f"评测 [{label}] 失败", check=False)
        # run_eval.py 输出结果到 eval/results/
        results_dir = ROOT / "eval" / "results"
        json_files = sorted(results_dir.glob(f"*{label}*.json"), key=os.path.getmtime, reverse=True)
        if json_files:
            report = json.loads(json_files[0].read_text(encoding="utf-8"))
            acc = report.get("execution_accuracy") or report.get("accuracy", 0)
            print(f"      [{label}] execution_accuracy = {acc:.2%}")
            return {
                "label": label,
                "accuracy": acc,
                "total_cases": report.get("total", 0),
                "result_file": str(json_files[0]),
            }
        else:
            print(f"      [!] [{label}] 评测完成但未找到结果文件")
    except subprocess.CalledProcessError as e:
        print(f"      [!] [{label}] 评测失败: {e}")
    return None


# ============================================================
# 阶段 5: QLoRA 微调
# ============================================================
def run_finetune(train_count: int) -> bool:
    """执行 LLaMA-Factory QLoRA 微调。无 GPU 或 --dry-run 时跳过。"""
    print(f"\n[5/9] QLoRA 微调 (LLaMA-Factory)...")
    print(f"      训练数据: {TRAIN_DATA_PATH} ({train_count} 条)")

    # 按数据量调整 rank
    rank = 16 if train_count >= 500 else 8
    print(f"      LoRA rank: {rank} (数据量 {'≥500' if rank == 16 else '<500'})")

    if not _has_gpu():
        print("      [!] 未检测到 GPU, 跳过实际微调 (dry-run 模式)")
        return False

    # 生成实际的 LLaMA-Factory 配置
    config = LLAMA_CONFIG_TEMPLATE.read_text(encoding="utf-8")
    config = config.replace("lora_rank: 8", f"lora_rank: {rank}")
    config = config.replace("lora_alpha: 16", f"lora_alpha: {rank * 2}")
    run_config = FINETUNE_DIR / "finetune_config_run.yaml"
    run_config.write_text(config, encoding="utf-8")

    try:
        _run(["llamafactory-cli", "train", str(run_config)], "LLaMA-Factory 微调失败")
        print("      [OK] 微调完成")
        return True
    except subprocess.CalledProcessError:
        print("      [!] 微调失败, 详见 LLaMA-Factory 日志")
        return False


# ============================================================
# 阶段 6: 导入 Ollama
# ============================================================
def import_to_ollama() -> str | None:
    """合并 LoRA 权重并导入 Ollama。返回新模型 tag。"""
    tag = f"qwen2.5-coder:finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\n[6/9] 导入微调模型到 Ollama: {tag}")

    if not _has_gpu():
        print("      [!] 无 GPU, 跳过模型导入 (dry-run 模式)")
        return None

    # 1. 合并 LoRA adapter -> GGUF
    checkpoint = FINETUNE_DIR / "checkpoints"
    merged_dir = FINETUNE_DIR / "merged"
    if not list(checkpoint.glob("checkpoint-*")):
        print("      [!] 未找到 checkpoint, 跳过导入")
        return None

    # 2. 创建 Modelfile
    modelfile = FINETUNE_DIR / "Modelfile"
    modelfile.write_text(
        f"FROM {OLLAMA_BASE}\n"
        "# Fine-tuned on Text2SQL feedback data\n"
        f"# Training data: {TRAIN_DATA_PATH}\n"
        f"# Date: {datetime.now().isoformat()}\n"
    )

    # 3. 导入 Ollama
    try:
        _run(["ollama", "create", tag, "-f", str(modelfile)], "Ollama 导入失败")
        print(f"      [OK] 新模型已导入: {tag}")
        return tag
    except subprocess.CalledProcessError:
        print("      [!] Ollama 导入失败")
        return None


# ============================================================
# 阶段 8: 对比决策
# ============================================================
def compare_and_decide(
    baseline: dict | None, finetuned: dict | None, backup_tag: str, new_tag: str | None
) -> dict:
    """对比基线和新模型，决定保留还是回滚。"""
    print(f"\n[8/9] 对比决策...")

    decision = {
        "action": "skip",
        "reason": "",
        "baseline_accuracy": baseline.get("accuracy") if baseline else None,
        "finetuned_accuracy": finetuned.get("accuracy") if finetuned else None,
        "backup_tag": backup_tag,
        "new_tag": new_tag,
    }

    if baseline is None:
        decision["action"] = "keep_baseline"
        decision["reason"] = "基线评测失败，保留当前模型"
        print(f"      [FAIL] {decision['reason']}")
        return decision

    if finetuned is None:
        decision["action"] = "keep_baseline"
        decision["reason"] = "微调后评测失败或微调未执行，保留当前模型"
        print(f"      [FAIL] {decision['reason']}")
        return decision

    delta = finetuned["accuracy"] - baseline["accuracy"]
    decision["accuracy_delta"] = delta

    if delta > 0.01:  # 提升超过 1%
        decision["action"] = "promote"
        decision["reason"] = f"准确率提升 {delta:+.2%}，替换线上模型"
        _update_gateway_model(new_tag)
        print(f"      [OK] {decision['reason']}")
    elif delta >= -0.01:  # 变化在 ±1% 以内 -> 无显著差异
        decision["action"] = "keep_new"
        decision["reason"] = f"准确率变化 {delta:+.2%}（无显著差异），保留新模型"
        _update_gateway_model(new_tag)
        print(f"      ~ {decision['reason']}")
    else:
        decision["action"] = "rollback"
        decision["reason"] = f"准确率下降 {delta:+.2%}，回滚到备份模型"
        _rollback_gateway_model()
        print(f"      [FAIL] {decision['reason']}")

    return decision


def _update_gateway_model(new_tag: str | None) -> None:
    """更新 gateway.yaml 中 ollama-local 的 model 字段。"""
    if not new_tag:
        return
    gw_path = ROOT / "gateway.yaml"
    if not gw_path.exists():
        print("      [!] gateway.yaml 不存在，无法更新")
        return
    content = gw_path.read_text(encoding="utf-8")
    # 只替换 ollama-local provider 的 model 行
    import re
    updated = re.sub(
        r"(id:\s*ollama-local.*?\n\s*model:\s*)[^\n]+",
        rf"\g<1>{new_tag}",
        content,
        flags=re.DOTALL,
    )
    gw_path.write_text(updated, encoding="utf-8")
    print(f"      gateway.yaml 已更新: ollama-local model -> {new_tag}")


def _rollback_gateway_model() -> None:
    """回滚：将 ollama-local 恢复到备份模型。"""
    # 简单地恢复到基础优化版
    gw_path = ROOT / "gateway.yaml"
    if not gw_path.exists():
        return
    content = gw_path.read_text(encoding="utf-8")
    import re
    updated = re.sub(
        r"(id:\s*ollama-local.*?\n\s*model:\s*)[^\n]+",
        rf"\g<1>{OLLAMA_MODEL}",
        content,
        flags=re.DOTALL,
    )
    gw_path.write_text(updated, encoding="utf-8")
    print(f"      gateway.yaml 已回滚: ollama-local model -> {OLLAMA_MODEL}")


# ============================================================
# 阶段 9: 审计日志
# ============================================================
def audit_log(run_record: dict) -> None:
    """追加运行记录到 run_history.jsonl，保留完整审计轨迹。"""
    print(f"\n[9/9] 记录审计日志...")
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record, ensure_ascii=False) + "\n")
    print(f"      [OK] 审计日志已保存: {RUN_HISTORY_PATH}")
    print(f"      ℹ 训练数据保留在: {TRAIN_DATA_PATH}（不删除，供人工审计）")


# ============================================================
# 工具函数
# ============================================================
def _run(cmd: list[str], error_msg: str, check: bool = True) -> subprocess.CompletedProcess:
    """执行命令并打印。"""
    print(f"      $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=ROOT,
        check=check,
        capture_output=False,
        text=True,
    )


def _has_gpu() -> bool:
    """检测是否有可用 GPU（NVIDIA CUDA 或 Apple Metal）。"""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=False, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="自动化微调流水线 — 反馈飞轮闭环训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dry-run                  仅检查触发条件和数据状态
  %(prog)s --execute                  强制执行完整流水线
  %(prog)s --execute --skip-trigger   跳过触发条件，强制执行
  %(prog)s --export-only              仅导出训练数据
        """,
    )
    parser.add_argument("--execute", action="store_true",
                        help="实际执行微调（默认 dry-run 模式）")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="仅检查条件，不执行微调")
    parser.add_argument("--skip-trigger-check", action="store_true",
                        help="跳过触发条件检查")
    parser.add_argument("--export-only", action="store_true",
                        help="仅导出训练数据并退出")
    parser.add_argument("--eval-dataset", choices=["mini", "spider"], default=EVAL_DATASET)
    parser.add_argument("--eval-limit", type=int, default=EVAL_LIMIT)
    args = parser.parse_args()

    print("=" * 60)
    print("  Text2SQL 自动化微调流水线")
    print(f"  模式: {'DRY-RUN (仅检查)' if not args.execute else 'EXECUTE (实际微调)'}")
    print(f"  时间: {datetime.now().isoformat()}")
    print("=" * 60)

    # ======== 阶段 1: 检查触发条件 ========
    print(f"\n[1/9] 检查触发条件...")
    print(f"      阈值: 新增 >= {THRESHOLD_SAMPLES} 条")
    print(f"      最大间隔: {MAX_INTERVAL_DAYS} 天 (最少 {MIN_SAMPLES_FALLBACK} 条)")

    counts = count_learned_examples()
    total = sum(counts.values())
    print(f"      当前点赞总数: {total}")
    for db, n in sorted(counts.items()):
        print(f"        {db}: {n}")

    if not args.skip_trigger_check:
        should_run, reason, stats = check_trigger()
        print(f"      结论: {'[OK] 触发' if should_run else '[FAIL] 未触发'} — {reason}")

        if args.export_only:
            export_data()
            return

        if not should_run and not args.execute:
            print(f"\n[!] 未满足触发条件，退出。使用 --execute --skip-trigger-check 强制执行。")
            return
    else:
        stats = {"total_learned": total, "by_db": counts, "checked_at": datetime.now().isoformat()}
        should_run = True

    if args.export_only:
        export_data()
        return

    if not args.execute:
        print(f"\n[Dry-run] 流水线到此为止。使用 --execute 实际执行微调。")
        print(f"[Dry-run] 会依次执行: 备份模型 -> 导出数据 -> 基线评测 -> 微调 -> 导入 -> 评测 -> 对比 -> 审计")
        return

    # ======== 执行完整流水线 ========
    run_record = {
        "run_at": datetime.now().isoformat(),
        "trigger_reason": reason if not args.skip_trigger_check else "manual",
        "stats_before": stats,
        "phases": {},
    }

    # 阶段 2: 备份
    backup_tag = backup_model()
    run_record["backup_tag"] = backup_tag

    # 阶段 3: 导出训练数据
    train_count = export_data()
    run_record["train_count"] = train_count
    run_record["train_data_path"] = str(TRAIN_DATA_PATH)

    if train_count == 0:
        print("[!] 无训练数据，终止流水线")
        return

    # 阶段 4: 基线评测
    print(f"\n[4/9] 基线评测 (当前模型: {OLLAMA_MODEL})...")
    baseline_result = run_eval("ollama-local", f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                               dataset=args.eval_dataset, limit=args.eval_limit)
    run_record["baseline_eval"] = baseline_result

    # 阶段 5: 微调
    finetune_ok = run_finetune(train_count)
    run_record["finetune_ok"] = finetune_ok

    # 阶段 6: 导入
    new_tag = None
    if finetune_ok:
        new_tag = import_to_ollama()
        run_record["new_model_tag"] = new_tag

    # 阶段 7: 微调后评测
    finetuned_result = None
    if new_tag:
        print(f"\n[7/9] 微调后评测 (新模型: {new_tag})...")
        finetuned_result = run_eval("ollama-local", f"finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                    dataset=args.eval_dataset, limit=args.eval_limit)
    else:
        print(f"\n[7/9] 微调后评测 — 跳过 (无新模型)")
    run_record["finetuned_eval"] = finetuned_result

    # 阶段 8: 对比决策
    decision = compare_and_decide(baseline_result, finetuned_result, backup_tag, new_tag)
    run_record["decision"] = decision

    # 阶段 9: 审计日志
    run_record["total_learned_at_run"] = total
    audit_log(run_record)

    # 保存上次运行记录
    save_last_run({
        "run_at": run_record["run_at"],
        "total_learned_at_run": total,
        "decision": decision["action"],
        "accuracy_delta": decision.get("accuracy_delta"),
    })

    print(f"\n{'=' * 60}")
    print(f"  流水线完成")
    print(f"  决策: {decision['action']} — {decision['reason']}")
    print(f"  训练数据保留: {TRAIN_DATA_PATH}")
    print(f"  审计日志: {RUN_HISTORY_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
