"""路由策略 (Routing strategies) —— 网关的核心可插拔点。

router 的唯一职责：给定 provider 列表 + 健康视图，**产出一个有序的候选执行顺序**。
它不发请求、不计成败 —— 那是 gateway.py 编排 + breaker 的事。把「选谁先试」与
「怎么试/失败怎么办」解耦，使新增策略只需实现一个 order() 方法。

内置策略：
    priority      按 priority 升序(默认)。配合 breaker 即「优先级 + 故障转移」。
    weighted      按 weight 降序 —— 高权重者优先(确定性，便于测试/审计)。
    round_robin   带游标轮转，跨调用均摊流量(游标加锁保证线程安全)。
    cost          按估算单位成本(price_in+price_out)升序，省钱优先。
    latency       按 breaker 记录的 p50 延迟升序，快的优先。

所有策略都会把**已熔断(OPEN)的 provider 排到末尾**而非直接删除：保留它们作为
「万不得已的最后尝试」，避免全部节点都熔断时无候选可用。
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from .models import ProviderSpec


@runtime_checkable
class HealthView(Protocol):
    """breaker 注册表暴露给 router 的只读视图。"""

    def is_available(self, provider_id: str) -> bool: ...
    def p50_latency(self, provider_id: str) -> float: ...


class RoutingStrategy(Protocol):
    def order(self, providers: list[ProviderSpec], health: HealthView) -> list[ProviderSpec]: ...


def _healthy_first(providers, health, key):
    """通用：先按 key 排序，再把熔断节点稳定地沉到末尾(保留为最后兜底)。"""
    ordered = sorted(providers, key=key)
    up = [p for p in ordered if health.is_available(p.id)]
    down = [p for p in ordered if not health.is_available(p.id)]
    return up + down


class PriorityStrategy:
    def order(self, providers, health):
        return _healthy_first(providers, health, key=lambda p: p.priority)


class WeightedStrategy:
    def order(self, providers, health):
        # 高权重优先；权重相同按 priority 兜底，保证确定性。
        return _healthy_first(providers, health, key=lambda p: (-p.weight, p.priority))


class CostStrategy:
    def order(self, providers, health):
        def unit_cost(p: ProviderSpec):
            pin = p.price_in_per_m if p.price_in_per_m is not None else float("inf")
            pout = p.price_out_per_m if p.price_out_per_m is not None else float("inf")
            return (pin + pout, p.priority)

        return _healthy_first(providers, health, key=unit_cost)


class LatencyStrategy:
    def order(self, providers, health):
        return _healthy_first(
            providers, health, key=lambda p: (health.p50_latency(p.id), p.priority)
        )


class RoundRobinStrategy:
    """跨调用轮转起点，把流量均摊到各 provider。

    游标随每次 order() 自增并对 provider 数取模，决定本次的轮转起点；熔断节点照例
    沉到末尾。游标用锁保护，多线程下不会错乱。
    """

    def __init__(self) -> None:
        self._cursor = 0
        self._lock = threading.Lock()

    def order(self, providers, health):
        base = sorted(providers, key=lambda p: p.priority)
        n = len(base)
        if n == 0:
            return []
        with self._lock:
            start = self._cursor % n
            self._cursor = (self._cursor + 1) % n
        rotated = base[start:] + base[:start]
        up = [p for p in rotated if health.is_available(p.id)]
        down = [p for p in rotated if not health.is_available(p.id)]
        return up + down


# 名字 → 策略实例(无状态的可共享；round_robin 每次 build 一个新实例以独立游标)。
_SHARED = {
    "priority": PriorityStrategy(),
    "weighted": WeightedStrategy(),
    "cost": CostStrategy(),
    "latency": LatencyStrategy(),
}


def build_router(name: str) -> RoutingStrategy:
    """按名字构造路由策略；未知名字回落到 priority。"""
    if name == "round_robin":
        return RoundRobinStrategy()
    return _SHARED.get(name, _SHARED["priority"])
