"""限流与配额 (Rate limiting & quota) —— 按角色保护后端 provider。

复用 governance 的 Principal 作为身份：principal.role 即限流 key，不另造账户体系。
两道闸，任一超限即拒绝(抛 RateLimitedError)：

    令牌桶 TokenBucket          —— 控 QPS(瞬时并发/突发)。桶以 qps 个/秒回填，
                                   每请求取 1 个；取不到即超速。
    滑动窗口 SlidingWindowQuota —— 控每分钟 token 总量(成本/配额)。请求前用
                                   估算 token 预检，请求后用真实 usage 校正。

为什么令牌桶 + 滑动窗口而非单一算法：QPS 关心「这一刻能不能发」，配额关心「过去
60 秒累计花了多少」——两者语义不同，分开实现各自最自然，也便于单独测试。
时钟可注入(clock)，测试免真 sleep。
"""

from __future__ import annotations

import threading
import time

from .models import RoleLimit


class RateLimitedError(RuntimeError):
    """请求被限流/配额拦截。携带角色与原因，便于上层返回友好提示。"""

    def __init__(self, role: str, reason: str):
        self.role = role
        self.reason = reason
        super().__init__(f"角色「{role}」{reason}")


class TokenBucket:
    """经典令牌桶：容量 = qps(允许等量突发)，回填速率 = qps 个/秒。"""

    def __init__(self, qps: float, clock=time.monotonic):
        self.rate = qps
        self.capacity = qps
        self._tokens = qps
        self._last = clock()
        self._clock = clock
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        if self.rate == float("inf"):
            return True
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


class SlidingWindowQuota:
    """60 秒滑动窗口内的 token 配额。

    allow(est) 检查「窗口内已用 + 预估」是否超限并先记入预估；commit(actual) 在
    请求返回后把预估替换为真实用量(差额回补/补扣)，使长期配额贴近真实成本。
    """

    def __init__(self, tokens_per_min: int, clock=time.monotonic):
        self.limit = tokens_per_min
        self.window = 60.0
        self._clock = clock
        self._lock = threading.Lock()
        self._events: list[tuple[float, int]] = []  # (ts, tokens)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        self._events = [(t, n) for (t, n) in self._events if t >= cutoff]

    def _used(self, now: float) -> int:
        self._evict(now)
        return sum(n for _, n in self._events)

    def allow(self, est_tokens: int) -> bool:
        if self.limit >= 2**62:
            return True
        with self._lock:
            now = self._clock()
            if self._used(now) + est_tokens > self.limit:
                return False
            self._events.append((now, est_tokens))  # 先按预估占位
            return True

    def commit(self, est_tokens: int, actual_tokens: int) -> None:
        """用真实用量校正最近一次预估占位。"""
        if self.limit >= 2**62:
            return
        with self._lock:
            now = self._clock()
            # 把最近一条预估替换为真实值(找最后一条等于 est 的占位)
            for i in range(len(self._events) - 1, -1, -1):
                if self._events[i][1] == est_tokens:
                    self._events[i] = (self._events[i][0], actual_tokens)
                    break
            else:
                self._events.append((now, actual_tokens))


class RoleLimiter:
    """把每个角色映射到一对(令牌桶, 滑动窗口)，按 RoleLimit 配置惰性创建。"""

    def __init__(self, limit_lookup, clock=time.monotonic):
        # limit_lookup: (role) -> RoleLimit | None  —— 通常是 providers.limit_for
        self._lookup = limit_lookup
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}
        self._quotas: dict[str, SlidingWindowQuota] = {}
        self._lock = threading.Lock()

    def _ensure(self, role: str, lim: RoleLimit):
        if role not in self._buckets:
            with self._lock:
                if role not in self._buckets:
                    self._buckets[role] = TokenBucket(lim.qps, self._clock)
                    self._quotas[role] = SlidingWindowQuota(lim.tokens_per_min, self._clock)
        return self._buckets[role], self._quotas[role]

    def check(self, role: str, est_tokens: int) -> None:
        """请求前调用：超 QPS 或超配额即抛 RateLimitedError。"""
        lim = self._lookup(role)
        if lim is None:  # 无限额配置 → 不限流(完全向后兼容)
            return
        bucket, quota = self._ensure(role, lim)
        if not bucket.try_acquire():
            raise RateLimitedError(role, f"请求过于频繁(超过 {lim.qps} QPS)")
        if not quota.allow(est_tokens):
            raise RateLimitedError(role, f"已超过每分钟 {lim.tokens_per_min} token 配额")

    def commit(self, role: str, est_tokens: int, actual_tokens: int) -> None:
        """请求后调用：用真实 token 用量校正配额窗口。"""
        if self._lookup(role) is None:
            return
        q = self._quotas.get(role)
        if q is not None:
            q.commit(est_tokens, actual_tokens)


def estimate_tokens(messages: list[dict]) -> int:
    """请求前的粗略 token 估算：约 4 字符/token，仅用于配额预检(commit 会校正)。"""
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, chars // 4)
