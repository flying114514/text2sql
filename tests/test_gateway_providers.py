"""P-G1 网关数据层测试：gateway.yaml 解析、${ENV} 展开、退化开关。无网络。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from text2sql.gateway import providers  # noqa: E402
from text2sql.gateway.models import ProviderSpec, RoleLimit  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_provider_caches():
    """每个用例前后清空 providers 的 lru_cache，避免把临时 gateway.yaml 的解析结果
    泄漏给其它测试(否则 gateway_enabled() 会在别处误判为已启用)。"""
    providers.reset_caches()
    yield
    providers.reset_caches()


SAMPLE = """
defaults:
  strategy: weighted
  timeout: 30
  max_retries: 1
providers:
  - id: deepseek
    base_url: https://api.deepseek.com
    api_key: ${MY_KEY}
    model: deepseek-chat
    priority: 10
    weight: 3
    price_in_per_m: 0.27
    price_out_per_m: 1.10
  - id: qwen
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: literal-key
    model: qwen-plus
    priority: 20
limits:
  default:
    qps: 5
    tokens_per_min: 100000
  admin:
    qps: 50
"""


def _point_to(monkeypatch, tmp_path: Path, text: str | None):
    """让 providers 模块读取一个临时 gateway.yaml(或没有文件)。"""
    f = tmp_path / "gateway.yaml"
    if text is not None:
        f.write_text(text, encoding="utf-8")
    monkeypatch.setattr(providers, "GATEWAY_FILE", f)
    providers.reset_caches()


def test_no_file_means_gateway_disabled(monkeypatch, tmp_path):
    _point_to(monkeypatch, tmp_path, None)  # 不写文件
    assert providers.gateway_enabled() is False
    assert providers.load_providers() == ()
    assert providers.load_limits() == {}
    assert providers.limit_for("admin") is None  # 无限额表 → 完全不限


def test_parses_providers_and_expands_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_KEY", "sk-secret-123")
    _point_to(monkeypatch, tmp_path, SAMPLE)

    specs = providers.load_providers()
    assert [s.id for s in specs] == ["deepseek", "qwen"]
    ds = specs[0]
    assert isinstance(ds, ProviderSpec)
    assert ds.api_key == "sk-secret-123"  # ${MY_KEY} 已展开
    assert ds.weight == 3 and ds.priority == 10
    assert ds.price_in_per_m == 0.27 and ds.price_out_per_m == 1.10
    # defaults 下放到未显式声明的字段
    assert ds.timeout == 30.0 and ds.max_retries == 1
    # 第二个 provider 用字面 key、缺省 weight/price
    qwen = specs[1]
    assert qwen.api_key == "literal-key" and qwen.weight == 1
    assert qwen.price_in_per_m is None
    assert providers.gateway_enabled() is True
    assert providers.default_strategy() == "weighted"


def test_limits_parsing_and_fallback(monkeypatch, tmp_path):
    _point_to(monkeypatch, tmp_path, SAMPLE)
    limits = providers.load_limits()
    assert isinstance(limits["default"], RoleLimit)
    assert limits["default"].qps == 5 and limits["default"].tokens_per_min == 100000
    # admin 未给 tokens_per_min → 视为不限(很大的数)
    assert limits["admin"].qps == 50 and limits["admin"].tokens_per_min >= 2**62
    # 精确命中 / default 兜底
    assert providers.limit_for("admin").qps == 50
    assert providers.limit_for("nobody").qps == 5  # 落到 default


def test_missing_required_field_raises(monkeypatch, tmp_path):
    bad = "providers:\n  - id: x\n    model: m\n"  # 缺 base_url/api_key
    _point_to(monkeypatch, tmp_path, bad)
    with pytest.raises(RuntimeError, match="base_url"):
        providers.load_providers()
