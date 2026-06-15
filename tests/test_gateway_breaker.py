"""P-G3 熔断器状态机测试：closed→open→half-open→closed，假时钟无 sleep。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.gateway.breaker import (  # noqa: E402
    CLOSED,
    HALF_OPEN,
    OPEN,
    BreakerRegistry,
    CircuitBreaker,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_closed_to_open_after_threshold():
    clk = FakeClock()
    cb = CircuitBreaker(fail_threshold=3, reset_timeout=30, clock=clk)
    assert cb.state == CLOSED and cb.allow()
    cb.record(ok=False)
    cb.record(ok=False)
    assert cb.state == CLOSED  # 还没到阈值
    cb.record(ok=False)  # 第 3 次 → 跳闸
    assert cb.state == OPEN
    assert cb.allow() is False  # 熔断期间被跳过


def test_open_to_half_open_after_timeout_then_recover():
    clk = FakeClock()
    cb = CircuitBreaker(fail_threshold=1, reset_timeout=30, clock=clk)
    cb.record(ok=False)  # 立刻 open
    assert cb.state == OPEN and not cb.allow()

    clk.advance(29)
    assert cb.allow() is False  # 还没到冷却时间

    clk.advance(2)  # 累计 31s > 30s
    assert cb.state == HALF_OPEN
    assert cb.allow() is True  # 放一个探测
    assert cb.allow() is False  # 半开只放一个

    cb.record(ok=True)  # 探测成功 → 恢复
    assert cb.state == CLOSED and cb.allow()


def test_half_open_failure_returns_to_open():
    clk = FakeClock()
    cb = CircuitBreaker(fail_threshold=1, reset_timeout=10, clock=clk)
    cb.record(ok=False)
    clk.advance(11)
    assert cb.state == HALF_OPEN and cb.allow()
    cb.record(ok=False)  # 探测又失败
    assert cb.state == OPEN


def test_success_resets_failure_count():
    cb = CircuitBreaker(fail_threshold=3)
    cb.record(ok=False)
    cb.record(ok=False)
    cb.record(ok=True)  # 清零
    cb.record(ok=False)
    cb.record(ok=False)
    assert cb.state == CLOSED  # 没连续到 3 次


def test_registry_health_view():
    clk = FakeClock()
    reg = BreakerRegistry(fail_threshold=1, reset_timeout=5, clock=clk)
    assert reg.is_available("a")  # 首次访问即创建，默认可用
    reg.get("a").record(ok=False)
    assert reg.is_available("a") is False
    reg.get("b").record(ok=True, latency_s=0.4)
    assert reg.p50_latency("b") == 0.4
    assert reg.p50_latency("never-seen") == float("inf")
