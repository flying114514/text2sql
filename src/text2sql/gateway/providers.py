"""Provider 注册表 (LLM Gateway provider registry).

解析项目根的 `gateway.yaml`：把 providers 段变成 ProviderSpec 列表、把 limits 段
变成 {role: RoleLimit}，并为每个 provider 懒加载一个 OpenAI 客户端。

设计要点：
  * 文件不存在 → load_providers() 返回 []，这是网关的**退化开关**：上层据此回到
    现有的「主 + 单兜底」单 provider 逻辑，保证零配置完全向后兼容。
  * ${ENV} 占位符复用 sources._expand_env(同 connections.yaml 的语义)，secrets
    不进 gateway.yaml(可安全提交 example)。
  * 解析结果用 @lru_cache，与 governance.load_policies / semantics 的缓存风格一致。
"""

from __future__ import annotations

import threading
from functools import lru_cache

from openai import OpenAI

from ..config import ROOT_DIR, settings
from ..sources import _expand_env
from .models import ProviderSpec, RoleLimit

GATEWAY_FILE = ROOT_DIR / "gateway.yaml"

# id -> OpenAI client。懒加载并缓存，避免每次调用都重建连接池(同 llm.py 的单例思路，
# 只是从单个 _client 扩展成按 provider 的字典)。
_clients: dict[str, OpenAI] = {}
_clients_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    """读取并解析 gateway.yaml，缺失时返回 {}(网关关闭)。"""
    if not GATEWAY_FILE.exists():
        return {}
    import yaml  # 局部 import：只有真的存在配置文件时才需要

    try:
        return yaml.safe_load(GATEWAY_FILE.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 — 配置坏了要给出清晰错误，而不是深埋在调用里
        raise RuntimeError(f"failed to parse {GATEWAY_FILE.name}: {e}") from e


def gateway_enabled() -> bool:
    """True 当且仅当存在 gateway.yaml 且至少配置了一个 provider。"""
    return bool(_load_raw().get("providers"))


def default_strategy() -> str:
    """gateway.yaml defaults.strategy，缺省 'priority'(优先级 + 故障转移)。"""
    return str((_load_raw().get("defaults") or {}).get("strategy") or "priority")


@lru_cache(maxsize=1)
def load_providers() -> tuple[ProviderSpec, ...]:
    """解析 providers 段为 ProviderSpec 元组(不可变，便于跨线程共享与缓存)。

    返回空 → 触发上层退化路径。每个 provider 的 base_url/api_key 都做 ${ENV} 展开。
    """
    raw = _load_raw()
    defaults = raw.get("defaults") or {}
    d_timeout = float(defaults.get("timeout", settings.llm_timeout))
    d_retries = int(defaults.get("max_retries", settings.llm_max_retries))

    out: list[ProviderSpec] = []
    for i, p in enumerate(raw.get("providers") or []):
        pid = str(p.get("id") or f"provider{i}").strip()
        base_url = _expand_env(str(p.get("base_url", "")).strip())
        api_key = _expand_env(str(p.get("api_key", "")).strip())
        model = str(p.get("model", "")).strip()
        if not (base_url and api_key and model):
            raise RuntimeError(
                f"gateway.yaml provider '{pid}' 缺少 base_url / api_key / model 之一"
            )
        out.append(
            ProviderSpec(
                id=pid,
                base_url=base_url,
                api_key=api_key,
                model=model,
                priority=int(p.get("priority", 100)),
                weight=max(1, int(p.get("weight", 1))),
                timeout=float(p.get("timeout", d_timeout)),
                max_retries=int(p.get("max_retries", d_retries)),
                price_in_per_m=_opt_float(p.get("price_in_per_m")),
                price_out_per_m=_opt_float(p.get("price_out_per_m")),
            )
        )
    return tuple(out)


@lru_cache(maxsize=1)
def load_limits() -> dict[str, RoleLimit]:
    """解析 limits 段为 {role: RoleLimit}。'default' 作为未列出角色的兜底。"""
    raw = _load_raw()
    out: dict[str, RoleLimit] = {}
    for role, lim in (raw.get("limits") or {}).items():
        lim = lim or {}
        out[str(role)] = RoleLimit(
            qps=float(lim.get("qps", 0)) or float("inf"),  # 0/缺省 = 不限
            tokens_per_min=int(lim.get("tokens_per_min", 0)) or 2**63,
        )
    return out


def limit_for(role: str | None) -> RoleLimit | None:
    """取某角色的限额：精确命中 > 'default' > None(完全不限)。"""
    limits = load_limits()
    if not limits:
        return None
    if role and role in limits:
        return limits[role]
    return limits.get("default")


def get_client(spec: ProviderSpec) -> OpenAI:
    """按 provider.id 懒加载并缓存 OpenAI 客户端(线程安全)。"""
    client = _clients.get(spec.id)
    if client is not None:
        return client
    with _clients_lock:
        client = _clients.get(spec.id)  # 双检锁
        if client is None:
            client = OpenAI(
                api_key=spec.api_key,
                base_url=spec.base_url,
                timeout=spec.timeout,
                max_retries=spec.max_retries,
            )
            _clients[spec.id] = client
    return client


def _opt_float(v) -> float | None:
    return None if v is None else float(v)


def reset_caches() -> None:
    """清空所有缓存 —— 仅供测试在改写配置/环境后调用。"""
    _load_raw.cache_clear()
    load_providers.cache_clear()
    load_limits.cache_clear()
    with _clients_lock:
        _clients.clear()
