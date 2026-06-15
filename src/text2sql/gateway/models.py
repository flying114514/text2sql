"""网关数据结构 (LLM Gateway data structures).

这里只放**纯数据**：provider 规格、角色限额、单次执行的全链路结果。把数据与
行为(router/breaker/limiter/gateway)分开，是为了让这些结构能被任意子模块 import
而不产生循环依赖 —— router 需要 ProviderSpec，limiter 需要 RoleLimit，但它们彼此
不需要知道对方。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    """一个可路由的 LLM provider。

    frozen=True：注册表一旦从 gateway.yaml 解析出来就不应被运行时改写(健康度/延迟
    等可变状态由 breaker 单独持有，而非塞进这里)，这样 ProviderSpec 可安全地在线程
    间共享、做字典 key、被策略反复读取。

    price_*_per_m 为 None 时回落到全局 settings.price_*_per_m —— 这是「按模型定价」
    对旧的「全局单价」的向后兼容点(见 pricing.estimate_cost_for)。
    """

    id: str  # 唯一标识，进 trace 和成本报表
    base_url: str
    api_key: str  # 已展开 ${ENV} 后的明文，仅在进程内存活
    model: str
    priority: int = 100  # 越小越优先(priority 策略用)
    weight: int = 1  # weighted / round_robin 分流权重
    timeout: float = 60.0
    max_retries: int = 2  # SDK 层对**单个** provider 的网络重试(与 breaker 正交)
    price_in_per_m: float | None = None  # USD / 百万 input token
    price_out_per_m: float | None = None  # USD / 百万 output token


@dataclass(frozen=True)
class RoleLimit:
    """某个角色的限流/配额额度。来自 gateway.yaml 的 limits 段。"""

    qps: float  # 每秒请求数上限(令牌桶速率)
    tokens_per_min: int  # 每 60s 滑动窗口内的 token 配额


@dataclass
class Attempt:
    """一次对单个 provider 的尝试记录 —— 成功或失败都留痕，构成 failover 全链路。"""

    provider_id: str
    model: str
    ok: bool
    latency_s: float
    error: str | None = None


@dataclass
class GatewayResult:
    """网关一次 complete() 的完整结果：最终响应 + 它是怎么得来的。

    attempts 让调用方/报表能看到「主 provider 失败、降级到次选才成功」这种过程，
    而不只是最终那一个响应 —— 这是网关相对裸 SDK 调用最有价值的可观测性增量。
    """

    response: object  # llm.LLMResponse；用 object 注解避免 gateway→llm 的导入环
    provider_id: str  # 最终成功的 provider
    attempts: list[Attempt] = field(default_factory=list)
