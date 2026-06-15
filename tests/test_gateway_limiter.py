"""P-G4 限流测试：令牌桶 QPS、滑动窗口配额、commit 校正、角色隔离。假时钟。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from text2sql.gateway.limiter import (  # noqa: E402
    RateLimitedError,
    RoleLimiter,
    SlidingWindowQuota,
    TokenBucket,
    estimate_tokens,
)
from text2sql.gateway.models import RoleLimit  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_token_bucket_blocks_burst_then_refills():
    clk = FakeClock()
    b = TokenBucket(qps=2, clock=clk)
    assert b.try_acquire() and b.try_acquire()  # 容量=2，两个突发 OK
    assert b.try_acquire() is False  # 第三个被挡
    clk.advance(0.5)  # 0.5s * 2/s = 1 个回填
    assert b.try_acquire() is True
    assert b.try_acquire() is False


def test_sliding_window_quota():
    clk = FakeClock()
    q = SlidingWindowQuota(tokens_per_min=100, clock=clk)
    assert q.allow(60)
    assert q.allow(30)  # 累计 90
    assert q.allow(20) is False  # 90+20 > 100
    clk.advance(61)  # 旧事件滑出窗口
    assert q.allow(80) is True


def test_quota_commit_corrects_estimate():
    clk = FakeClock()
    q = SlidingWindowQuota(tokens_per_min=100, clock=clk)
    assert q.allow(10)  # 预估 10
    q.commit(10, 90)  # 真实 90
    assert q.allow(15) is False  # 90+15 > 100，证明 commit 把用量提到了 90


def test_role_limiter_isolates_roles():
    limits = {
        "tight": RoleLimit(qps=1, tokens_per_min=1000),
        "loose": RoleLimit(qps=100, tokens_per_min=10**9),
    }
    clk = FakeClock()
    rl = RoleLimiter(lambda r: limits.get(r), clock=clk)
    rl.check("tight", 10)  # 第一次 OK
    with pytest.raises(RateLimitedError) as ei:
        rl.check("tight", 10)  # 同角色第二次超 1 QPS
    assert ei.value.role == "tight"
    rl.check("loose", 10)  # 另一角色不受影响


def test_no_limit_config_means_unlimited():
    rl = RoleLimiter(lambda r: None)  # 无限额表
    for _ in range(100):
        rl.check("anyone", 999999)  # 永不抛错


def test_quota_exceeded_raises():
    limits = {"r": RoleLimit(qps=1000, tokens_per_min=50)}
    rl = RoleLimiter(lambda r: limits.get(r))
    with pytest.raises(RateLimitedError, match="配额"):
        rl.check("r", 80)  # 一次就超配额


def test_estimate_tokens():
    assert estimate_tokens([{"content": "a" * 400}]) == 100
    assert estimate_tokens([{"content": ""}]) == 1  # 下限
