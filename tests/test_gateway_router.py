"""P-G3 路由策略测试：五种策略排序 + 熔断节点沉底 + round_robin 轮转。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.gateway.models import ProviderSpec  # noqa: E402
from text2sql.gateway.router import build_router  # noqa: E402


def spec(id, priority=100, weight=1, pin=None, pout=None):
    return ProviderSpec(
        id=id,
        base_url="x",
        api_key="k",
        model="m",
        priority=priority,
        weight=weight,
        price_in_per_m=pin,
        price_out_per_m=pout,
    )


class FakeHealth:
    """可控的健康视图：down 集合里的视为熔断，latency 映射给 latency 策略。"""

    def __init__(self, down=(), latency=None):
        self.down = set(down)
        self.latency = latency or {}

    def is_available(self, pid):
        return pid not in self.down

    def p50_latency(self, pid):
        return self.latency.get(pid, float("inf"))


def ids(specs):
    return [s.id for s in specs]


def test_priority_strategy():
    ps = [spec("c", 30), spec("a", 10), spec("b", 20)]
    r = build_router("priority")
    assert ids(r.order(ps, FakeHealth())) == ["a", "b", "c"]


def test_weighted_strategy_prefers_high_weight():
    ps = [spec("a", 10, weight=1), spec("b", 20, weight=5)]
    r = build_router("weighted")
    assert ids(r.order(ps, FakeHealth())) == ["b", "a"]


def test_cost_strategy_cheapest_first():
    ps = [
        spec("pricey", 10, pin=5.0, pout=5.0),
        spec("cheap", 20, pin=0.1, pout=0.2),
    ]
    r = build_router("cost")
    assert ids(r.order(ps, FakeHealth())) == ["cheap", "pricey"]


def test_latency_strategy_fastest_first():
    ps = [spec("slow", 10), spec("fast", 20)]
    health = FakeHealth(latency={"slow": 2.0, "fast": 0.3})
    r = build_router("latency")
    assert ids(r.order(ps, health)) == ["fast", "slow"]


def test_open_breaker_sinks_to_bottom():
    ps = [spec("a", 10), spec("b", 20), spec("c", 30)]
    r = build_router("priority")
    # a 熔断 → 仍保留但排到最后(全挂时还能兜底)
    out = ids(r.order(ps, FakeHealth(down={"a"})))
    assert out == ["b", "c", "a"]


def test_round_robin_rotates_across_calls():
    ps = [spec("a", 10), spec("b", 20), spec("c", 30)]
    r = build_router("round_robin")
    h = FakeHealth()
    first = ids(r.order(ps, h))[0]
    second = ids(r.order(ps, h))[0]
    third = ids(r.order(ps, h))[0]
    assert [first, second, third] == ["a", "b", "c"]  # 起点逐次轮转


def test_unknown_strategy_falls_back_to_priority():
    ps = [spec("b", 20), spec("a", 10)]
    r = build_router("nonsense")
    assert ids(r.order(ps, FakeHealth())) == ["a", "b"]
