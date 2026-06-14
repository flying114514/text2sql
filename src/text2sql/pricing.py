"""Cost estimation — one place so tracing and the eval agree on numbers.

Prices are configurable (per provider/plan) via PRICE_IN/OUT in .env and are
always labelled an *estimate* wherever they surface.
"""

from __future__ import annotations

from .config import settings


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1e6 * settings.price_in_per_m
        + completion_tokens / 1e6 * settings.price_out_per_m
    )
