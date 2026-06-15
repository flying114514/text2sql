"""熔断器 (Circuit Breaker) —— provider 级故障隔离。

为什么需要：当某个 provider 挂了，最朴素的故障转移每次请求都会先打它一次、超时、
再降级 —— 故障节点持续拖慢每一个请求。熔断器记住「这个节点最近一直在失败」，在
冷却期内直接跳过它，冷却后再放一个探测请求试探恢复。

三态机(每个 provider 一个 CircuitBreaker，线程安全)：
    closed     正常放行。连续失败累计到 fail_threshold → 转 open。
    open       熔断。allow() 返回 False，router 跳过该 provider。
               距进入 open 超过 reset_timeout 秒 → 转 half-open。
    half-open   半开探测。只放行**一个**请求：成功 → closed(恢复)；失败 → 回 open。

时钟通过构造注入(clock)，测试用假时钟即可验证 reset_timeout 边界而无需真 sleep。
"""

from __future__ import annotations

import threading
import time
from collections import deque

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        fail_threshold: int = 3,
        reset_timeout: float = 30.0,
        clock=time.monotonic,
    ) -> None:
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probing = False  # half-open 期间是否已放出探测请求
        self._latencies: deque[float] = deque(maxlen=50)  # 给 LatencyStrategy 估 p50

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def allow(self) -> bool:
        """router 在选中某 provider 前调用：open 期间(未到冷却)返回 False 以跳过。"""
        with self._lock:
            self._maybe_half_open()
            if self._state == OPEN:
                return False
            if self._state == HALF_OPEN:
                # 半开态只允许一个探测请求在途，其余照旧跳过
                if self._probing:
                    return False
                self._probing = True
            return True

    def record(self, ok: bool, latency_s: float = 0.0) -> None:
        """请求结束后回报结果，驱动状态转移。"""
        with self._lock:
            if ok:
                self._latencies.append(latency_s)
                self._consecutive_failures = 0
                self._state = CLOSED  # 成功(含 half-open 探测成功)→ 恢复
                self._probing = False
            else:
                self._consecutive_failures += 1
                self._probing = False
                if self._state == HALF_OPEN:
                    self._trip()  # 探测又失败 → 立刻回 open
                elif self._consecutive_failures >= self.fail_threshold:
                    self._trip()

    def p50_latency(self) -> float:
        """最近成功请求的中位延迟(无数据返回 inf，让 LatencyStrategy 视其为最差)。"""
        with self._lock:
            if not self._latencies:
                return float("inf")
            s = sorted(self._latencies)
            return s[len(s) // 2]

    # --- internals (调用方已持锁) ---
    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self._clock()

    def _maybe_half_open(self) -> None:
        if self._state == OPEN and (self._clock() - self._opened_at) >= self.reset_timeout:
            self._state = HALF_OPEN
            self._probing = False


class BreakerRegistry:
    """provider id → CircuitBreaker。同时充当 router 看到的 HealthView。"""

    def __init__(
        self, *, fail_threshold: int = 3, reset_timeout: float = 30.0, clock=time.monotonic
    ):
        self._cfg = {
            "fail_threshold": fail_threshold,
            "reset_timeout": reset_timeout,
            "clock": clock,
        }
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, provider_id: str) -> CircuitBreaker:
        b = self._breakers.get(provider_id)
        if b is not None:
            return b
        with self._lock:
            b = self._breakers.get(provider_id)
            if b is None:
                b = CircuitBreaker(**self._cfg)
                self._breakers[provider_id] = b
        return b

    # --- HealthView：router 据此跳过熔断节点 / 读延迟 ---
    def is_available(self, provider_id: str) -> bool:
        return self.get(provider_id).state != OPEN

    def p50_latency(self, provider_id: str) -> float:
        return self.get(provider_id).p50_latency()
