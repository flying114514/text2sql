"""P-G2 定价升级测试：按模型定价 + 全局回落 + 旧 API 兼容。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql import pricing  # noqa: E402
from text2sql.pricing import estimate_cost, estimate_cost_for  # noqa: E402


def test_global_price_backward_compatible(monkeypatch):
    monkeypatch.setattr(pricing.settings, "price_in_per_m", 1.0)
    monkeypatch.setattr(pricing.settings, "price_out_per_m", 2.0)
    # 1M input @1.0 + 0.5M output @2.0 = 1.0 + 1.0 = 2.0
    assert estimate_cost(1_000_000, 500_000) == 2.0
    # 新函数无 override 时与旧函数完全一致
    assert estimate_cost_for(1_000_000, 500_000) == estimate_cost(1_000_000, 500_000)


def test_per_model_price_override(monkeypatch):
    monkeypatch.setattr(pricing.settings, "price_in_per_m", 1.0)
    monkeypatch.setattr(pricing.settings, "price_out_per_m", 2.0)
    # 用模型自己的价(0.27 / 1.10)，不受全局影响
    cost = estimate_cost_for(1_000_000, 1_000_000, price_in=0.27, price_out=1.10)
    assert abs(cost - (0.27 + 1.10)) < 1e-9


def test_partial_override_falls_back(monkeypatch):
    monkeypatch.setattr(pricing.settings, "price_in_per_m", 5.0)
    monkeypatch.setattr(pricing.settings, "price_out_per_m", 9.0)
    # 只覆盖 input 价，output 回落全局 9.0
    cost = estimate_cost_for(1_000_000, 1_000_000, price_in=0.5)
    assert abs(cost - (0.5 + 9.0)) < 1e-9


def test_tracing_event_carries_provider_and_costs(monkeypatch, tmp_path):
    from text2sql import tracing

    monkeypatch.setattr(tracing, "TRACES_DIR", tmp_path)
    event = tracing.record_llm_call(
        kind="completion",
        provider="deepseek",
        model="deepseek-chat",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        latency_s=0.1,
        price_in=0.27,
        price_out=1.10,
    )
    assert event["provider"] == "deepseek"
    assert abs(event["cost_usd"] - 0.27) < 1e-6
    # 旧式调用(不传 provider)仍可用，provider 为 None
    legacy = tracing.record_llm_call(
        kind="completion",
        model="m",
        prompt_tokens=0,
        completion_tokens=0,
        latency_s=0.0,
    )
    assert legacy["provider"] is None
