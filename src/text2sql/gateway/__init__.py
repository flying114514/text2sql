"""LLM Gateway —— 进程内大模型网关。

对外门面在 gateway.py(P-G5 接入)。本包内的子模块职责：
  models     纯数据结构(ProviderSpec / RoleLimit / GatewayResult)
  providers  解析 gateway.yaml → provider 注册表 + 限额 + 客户端
  router     可插拔路由策略(优先级/加权/轮询/成本/延迟)
  breaker    每 provider 的三态熔断器
  limiter    按角色的令牌桶限流 + token 配额
  gateway    编排门面：limiter → router → breaker → 调用 → trace → 故障转移
"""

from __future__ import annotations
