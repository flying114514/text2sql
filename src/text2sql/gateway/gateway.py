"""网关编排门面 (Gateway facade)。

把四个子系统串成一次有韧性的 complete()：

    limiter.check(role)          按角色限流/配额，超限直接拒绝
        ↓
    router.order(specs, health)  选出候选执行顺序(跳过/沉底熔断节点)
        ↓ 逐个候选
    breaker.allow(id)            熔断节点跳过
        ↓
    llm._call(client, model)     复用现有唯一 raw 调用点(测试在此 mock)
        ↓
    breaker.record + trace       记成败/延迟/成本(provider 维度)
        ↓ 成功即返回；失败转下一候选
    全部失败 → AllProvidersFailedError(带 attempts 全链路)

复用而非重写：raw 调用走 llm._call，定价/落盘走 tracing.record_llm_call，身份用
governance.Principal。网关只负责「选谁、何时熔断、谁来限流」的编排。
"""

from __future__ import annotations

import time

from .breaker import BreakerRegistry
from .limiter import RoleLimiter, estimate_tokens
from .models import Attempt, GatewayResult
from .providers import default_strategy, get_client, limit_for, load_providers
from .router import build_router


class AllProvidersFailedError(RuntimeError):
    """所有候选 provider 都失败。attempts 记录每一跳，便于诊断/审计。"""

    def __init__(self, attempts: list[Attempt]):
        self.attempts = attempts
        chain = " → ".join(f"{a.provider_id}({a.error})" for a in attempts if not a.ok)
        super().__init__(f"all providers failed: {chain}")


# 进程级单例：熔断状态与限流计数需要跨请求累积，所以不能每次新建。
_breakers = BreakerRegistry()
_limiter = RoleLimiter(limit_for)
_router = None
_router_name = None


def _get_router():
    """按配置的策略名构造 router，并在策略名变化时重建(便于测试切换)。"""
    global _router, _router_name
    name = default_strategy()
    if _router is None or name != _router_name:
        _router = build_router(name)
        _router_name = name
    return _router


def complete(messages, *, temperature, json_mode, principal=None, provider_id=None) -> GatewayResult:
    """经网关发起一次补全。调用方应保证 load_providers() 非空(否则走退化路径)。

    provider_id 不为空时只尝试该 provider（供前端模型切换按钮直接指定）。
    """
    from .. import llm  # 延迟导入，打破 gateway ↔ llm 的导入环

    role = getattr(principal, "role", None)
    est = estimate_tokens(messages)
    if role is not None:
        _limiter.check(role, est)  # 超限抛 RateLimitedError，不消耗任何 provider

    specs = list(load_providers())
    if provider_id:
        specs = [s for s in specs if s.id == provider_id]
        if not specs:
            raise RuntimeError(f"provider '{provider_id}' not found in gateway.yaml")
    order = _get_router().order(specs, _breakers)

    attempts: list[Attempt] = []
    for spec in order:
        breaker = _breakers.get(spec.id)
        if not breaker.allow():
            continue  # 熔断中且非探测窗口
        t0 = time.perf_counter()
        try:
            resp = llm._call(get_client(spec), spec.model, messages, temperature, json_mode)
            latency = time.perf_counter() - t0
            breaker.record(ok=True, latency_s=latency)
            attempts.append(Attempt(spec.id, spec.model, True, latency))
            _trace(spec, resp, latency, ok=True)
            if role is not None:
                _limiter.commit(role, est, resp.total_tokens)
            return GatewayResult(response=resp, provider_id=spec.id, attempts=attempts)
        except Exception as e:  # noqa: BLE001 — 任何失败都转下一候选
            latency = time.perf_counter() - t0
            breaker.record(ok=False, latency_s=latency)
            err = f"{type(e).__name__}: {e}"
            attempts.append(Attempt(spec.id, spec.model, False, latency, err))
            _trace(spec, None, latency, ok=False, error=err)
            continue

    raise AllProvidersFailedError(attempts)


def _trace(spec, resp, latency, *, ok, error=None):
    from ..tracing import record_llm_call

    record_llm_call(
        kind="completion",
        provider=spec.id,
        model=spec.model,
        prompt_tokens=getattr(resp, "prompt_tokens", 0) if resp else 0,
        completion_tokens=getattr(resp, "completion_tokens", 0) if resp else 0,
        latency_s=latency,
        ok=ok,
        error=error,
        price_in=spec.price_in_per_m,
        price_out=spec.price_out_per_m,
    )


def reset_state() -> None:
    """重置熔断/限流/路由单例 —— 仅供测试。"""
    global _breakers, _limiter, _router, _router_name
    _breakers = BreakerRegistry()
    _limiter = RoleLimiter(limit_for)
    _router = None
    _router_name = None


def runtime_status() -> dict:
    """网关运行态快照(只读) —— 供监控页展示「谁被熔断了 / 当前策略 / 限额」。

    纯读取:不发任何请求、不新建单例。某 provider 若还没被调用过,其熔断器尚未
    惰性创建 —— 此时视为 closed(健康),而不是去 get() 把它建出来(避免观测产生副作用)。
    """
    from .providers import gateway_enabled, load_limits

    if not gateway_enabled():
        return {"enabled": False, "strategy": None, "providers": [], "limits": {}}

    seen = _breakers._breakers  # 已惰性创建的熔断器(只读 peek,不触发创建)
    providers = []
    for spec in load_providers():
        br = seen.get(spec.id)
        providers.append(
            {
                "id": spec.id,
                "model": spec.model,
                "priority": spec.priority,
                "weight": spec.weight,
                "state": br.state if br is not None else "closed",
                "p50_latency_s": (
                    None if br is None or br.p50_latency() == float("inf") else br.p50_latency()
                ),
                "price_in_per_m": spec.price_in_per_m,
                "price_out_per_m": spec.price_out_per_m,
            }
        )

    limits = {
        role: {
            "qps": None if lim.qps == float("inf") else lim.qps,
            "tokens_per_min": None if lim.tokens_per_min >= 2**62 else lim.tokens_per_min,
        }
        for role, lim in load_limits().items()
    }
    return {
        "enabled": True,
        "strategy": default_strategy(),
        "providers": providers,
        "limits": limits,
    }
