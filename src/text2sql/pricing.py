"""Cost estimation — one place so tracing and the eval agree on numbers.

Prices are configurable (per provider/plan) via PRICE_IN/OUT in .env and are
always labelled an *estimate* wherever they surface.

Two layers, smallest-surface first:
  * estimate_cost(p, c)      — the original global-price API (unchanged).
  * estimate_cost_for(...)   — per-model price override for the gateway, falling
                               back to the global price when a model has none.
"""

from __future__ import annotations

from .config import settings


def estimate_cost_for(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    price_in: float | None = None,
    price_out: float | None = None,
) -> float:
    """Cost using per-model prices when given, else the global PRICE_IN/OUT.

    The gateway passes a provider's own price_*_per_m here so multi-model traces
    cost each call correctly; legacy callers use estimate_cost() and get the
    single global rate exactly as before.
    """
    pin = settings.price_in_per_m if price_in is None else price_in
    pout = settings.price_out_per_m if price_out is None else price_out
    return prompt_tokens / 1e6 * pin + completion_tokens / 1e6 * pout


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Global-price cost. Kept for backward compatibility — delegates to the
    per-model function with no overrides."""
    return estimate_cost_for(prompt_tokens, completion_tokens)
