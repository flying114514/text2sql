"""全链路观测 (Phase 5 + Langfuse v4).

Langfuse v4 使用 OpenTelemetry + @observe() 装饰器模式，自动创建 Trace→Span→Generation 层级:
  - @observe() 最外层 → Trace
  - @observe() 内层嵌套 → Span
  - @observe(as_type="generation") → LLM Generation

三层设计:
  1. 本地 JSONL  — 始终开启，零依赖，按天归档到 data/traces/
  2. Langfuse   — .env 配置三行即可开启，@observe() 自动上报
  3. 网关监控    — /api/gateway/metrics 提供 provider 级别聚合

Trace 结构（自动生成）:
  answer_query ("各城市GMV")
    ├── converse ("llm_generate")
    │   └── _call ("completion") [generation] → model + tokens + latency + cost
    └── execute_sql
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from .config import DATA_DIR

TRACES_DIR = DATA_DIR / "traces"
_lock = threading.Lock()


def langfuse_available() -> bool:
    """LangFuse 是否已配置并可用。"""
    from .config import settings
    return settings.langfuse_ready()


# ============================================================
# 本地 JSONL 记录（始终开启）
# ============================================================
def _write_local(event: dict) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = TRACES_DIR / f"llm-{day}.jsonl"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ============================================================
# LLM 调用记录（核心观测点）
# ============================================================
def record_llm_call(
    *,
    kind: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    ok: bool = True,
    error: str | None = None,
    provider: str | None = None,
    price_in: float | None = None,
    price_out: float | None = None,
) -> dict:
    """记录一次 LLM 调用。本地 JSONL 始终写入；Langfuse 由 @observe 装饰器自动处理。"""
    from .pricing import estimate_cost_for

    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_s": round(latency_s, 4),
        "cost_usd": round(
            estimate_cost_for(prompt_tokens, completion_tokens, price_in=price_in, price_out=price_out),
            6,
        ),
        "ok": ok,
        "error": error,
    }
    _write_local(event)
    return event


# ============================================================
# 用户反馈打分（👍/👎 → Langfuse score）
# ============================================================
def record_feedback(question: str, rating: str, trace_id: str | None = None) -> None:
    """将用户 👍/👎 作为 score 上报到 Langfuse。"""
    if not langfuse_available():
        return
    try:
        from langfuse import get_client
        client = get_client()
        value = 1.0 if rating == "up" else 0.0
        client.create_score(
            trace_id=trace_id,
            name="user_feedback",
            value=value,
            comment=question,
        )
    except Exception:
        pass
