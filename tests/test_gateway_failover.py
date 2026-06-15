"""P-G5 网关编排 + 接入测试：故障转移、限流拦截、全失败、退化兼容。无网络。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from text2sql import llm  # noqa: E402
from text2sql.gateway import gateway, providers  # noqa: E402
from text2sql.gateway.gateway import AllProvidersFailedError  # noqa: E402
from text2sql.gateway.limiter import RateLimitedError  # noqa: E402
from text2sql.gateway.models import ProviderSpec, RoleLimit  # noqa: E402
from text2sql.llm import LLMResponse  # noqa: E402


def _two_providers():
    return (
        ProviderSpec(id="a", base_url="x", api_key="k", model="ma", priority=10),
        ProviderSpec(id="b", base_url="x", api_key="k", model="mb", priority=20),
    )


def _setup(monkeypatch, specs, limits=None, strategy="priority"):
    """绕过文件 IO，直接注入 provider/limit/strategy，并重置网关单例。"""
    monkeypatch.setattr(providers, "load_providers", lambda: specs)
    monkeypatch.setattr(providers, "default_strategy", lambda: strategy)
    monkeypatch.setattr(providers, "limit_for", lambda role: (limits or {}).get(role))
    # gateway 模块在 import 时已绑定了这些名字，需同步打补丁
    monkeypatch.setattr(gateway, "load_providers", lambda: specs)
    monkeypatch.setattr(gateway, "default_strategy", lambda: strategy)
    monkeypatch.setattr(gateway, "limit_for", lambda role: (limits or {}).get(role))
    monkeypatch.setattr(gateway, "get_client", lambda spec: object())
    gateway.reset_state()


def test_failover_primary_fails_secondary_succeeds(monkeypatch):
    specs = _two_providers()
    _setup(monkeypatch, specs)

    calls = []

    def fake_call(client, model, messages, temperature, json_mode):
        calls.append(model)
        if model == "ma":
            raise RuntimeError("boom")
        return LLMResponse(content="ok", prompt_tokens=1, completion_tokens=2, model=model)

    monkeypatch.setattr(llm, "_call", fake_call)

    result = gateway.complete([{"role": "user", "content": "hi"}], temperature=0.0, json_mode=False)
    assert result.response.content == "ok"
    assert result.provider_id == "b"
    assert calls == ["ma", "mb"]  # 先试主、失败后降级
    assert [a.ok for a in result.attempts] == [False, True]


def test_all_providers_fail_raises(monkeypatch):
    _setup(monkeypatch, _two_providers())
    monkeypatch.setattr(llm, "_call", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(AllProvidersFailedError) as ei:
        gateway.complete([{"role": "user", "content": "x"}], temperature=0.0, json_mode=False)
    assert len(ei.value.attempts) == 2 and all(not a.ok for a in ei.value.attempts)


def test_breaker_opens_and_skips_after_repeated_failure(monkeypatch):
    # 单 provider，连续失败到阈值后第三次直接被熔断跳过(不再调用)
    specs = (ProviderSpec(id="a", base_url="x", api_key="k", model="ma"),)
    _setup(monkeypatch, specs)
    n = {"calls": 0}

    def fail(*a, **k):
        n["calls"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(llm, "_call", fail)
    for _ in range(3):  # 默认 fail_threshold=3
        try:
            gateway.complete([{"content": "x"}], temperature=0.0, json_mode=False)
        except AllProvidersFailedError:
            pass
    before = n["calls"]
    # 此时 breaker 已 open，下一次没有候选可放行
    try:
        gateway.complete([{"content": "x"}], temperature=0.0, json_mode=False)
    except AllProvidersFailedError:
        pass
    assert n["calls"] == before  # 熔断后未再真正调用 provider


def test_rate_limit_blocks_before_calling_provider(monkeypatch):
    specs = _two_providers()
    limits = {"tight": RoleLimit(qps=1, tokens_per_min=10**9)}
    _setup(monkeypatch, specs, limits=limits)
    monkeypatch.setattr(
        llm,
        "_call",
        lambda *a, **k: LLMResponse(content="ok", prompt_tokens=1, completion_tokens=1),
    )

    class P:
        role = "tight"

    msgs = [{"role": "user", "content": "hello"}]
    gateway.complete(msgs, temperature=0.0, json_mode=False, principal=P())  # 第一次 OK
    with pytest.raises(RateLimitedError) as ei:
        gateway.complete(msgs, temperature=0.0, json_mode=False, principal=P())  # 超 1 QPS
    assert ei.value.role == "tight"


# --- 退化兼容：无 gateway.yaml 时 llm.complete 走旧路径 ---------------------
def test_llm_complete_legacy_when_gateway_disabled(monkeypatch):
    monkeypatch.setattr("text2sql.gateway.providers.gateway_enabled", lambda: False)
    monkeypatch.setattr(llm.settings, "llm_model", "legacy-model")
    monkeypatch.setattr(llm.settings, "fallback_model", "")
    seen = {}

    def fake_call(client, model, messages, temperature, json_mode):
        seen["model"] = model
        return LLMResponse(content="legacy-ok", prompt_tokens=1, completion_tokens=1, model=model)

    monkeypatch.setattr(llm, "_call", fake_call)
    monkeypatch.setattr(llm, "get_client", lambda: object())

    resp = llm.complete([{"role": "user", "content": "hi"}])
    assert resp.content == "legacy-ok"
    assert seen["model"] == "legacy-model"  # 证明走的是单 provider 旧路径


def test_llm_complete_delegates_when_gateway_enabled(monkeypatch):
    monkeypatch.setattr("text2sql.gateway.providers.gateway_enabled", lambda: True)
    captured = {}

    class FakeResult:
        response = LLMResponse(content="via-gateway", prompt_tokens=0, completion_tokens=0)

    def fake_gw_complete(messages, *, temperature, json_mode, principal=None):
        captured["used"] = True
        return FakeResult()

    monkeypatch.setattr("text2sql.gateway.gateway.complete", fake_gw_complete)
    resp = llm.complete([{"content": "x"}])
    assert resp.content == "via-gateway" and captured["used"]
