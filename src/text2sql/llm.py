"""Thin LLM client wrapper around the OpenAI-compatible SDK.

Why a wrapper instead of calling the SDK directly everywhere:
  * one place to inject base_url / api_key / timeout / retries from settings,
  * one place to later add observability (Phase 5) and model fallback,
  * the rest of the codebase depends on *our* small interface, not the vendor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from .config import settings
from .tracing import record_llm_call


@dataclass
class LLMResponse:
    """A completion plus the token usage it cost — needed for evaluation."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Built lazily so that merely importing this module doesn't require a key.
_client: OpenAI | None = None
_fallback_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings.assert_ready()
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    return _client


def get_fallback_client() -> OpenAI:
    """Client for the fallback model (reuses primary creds if not overridden)."""
    global _fallback_client
    if _fallback_client is None:
        _fallback_client = OpenAI(
            api_key=settings.fallback_api_key or settings.llm_api_key,
            base_url=settings.fallback_base_url or settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    return _fallback_client


def _call(
    client: OpenAI, model: str, messages: list[dict], temperature: float, json_mode: bool
) -> LLMResponse:
    """One raw model call. Isolated so it's the single point to mock in tests."""
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    usage = resp.usage
    return LLMResponse(
        content=resp.choices[0].message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model=model,
    )


def complete(
    messages: list[dict],
    *,
    temperature: float | None = None,
    json_mode: bool = False,
) -> LLMResponse:
    """Send a chat completion: trace it, and fall back to a backup model on error.

    Every attempt (success or failure) is recorded by the tracing layer, so token
    usage, latency and cost are auditable. If the primary model raises after the
    SDK's own retries, a configured fallback model is tried — this is the
    reliability layer that keeps the agent serving when one provider has a wobble.
    """
    temp = settings.llm_temperature if temperature is None else temperature

    attempts: list[tuple[OpenAI, str]] = [(get_client(), settings.llm_model)]
    if settings.fallback_ready():
        attempts.append((get_fallback_client(), settings.fallback_model))

    last_err: Exception | None = None
    for client, model in attempts:
        t0 = time.perf_counter()
        try:
            resp = _call(client, model, messages, temp, json_mode)
            record_llm_call(
                kind="completion",
                model=model,
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                latency_s=time.perf_counter() - t0,
                ok=True,
            )
            return resp
        except Exception as e:  # noqa: BLE001
            record_llm_call(
                kind="completion",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_s=time.perf_counter() - t0,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )
            last_err = e
            continue

    raise last_err if last_err else RuntimeError("no LLM attempt was made")


def chat(
    messages: list[dict],
    *,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Convenience wrapper returning just the text content."""
    return complete(messages, temperature=temperature, json_mode=json_mode).content


def ping() -> str:
    """Tiny health-check used by the Phase 0 smoke test."""
    return chat(
        [{"role": "user", "content": "Reply with exactly the word: pong"}],
        temperature=0.0,
    )
