"""OpenAI-compatible embedding client with an on-disk cache.

Only used by the *semantic* schema retriever. It is optional: if no embedding
endpoint is configured (EMBED_* in .env), the retriever falls back to the
lexical strategy and this module is never touched.

Why a disk cache: table schemas and eval questions are stable across runs, so
re-embedding them every evaluation would waste money and time. We key the cache
by a hash of (model, text) and persist it as a single JSON file, so a second
run is effectively free.
"""

from __future__ import annotations

import hashlib
import json
import threading

from openai import OpenAI

from .config import DATA_DIR, settings

_CACHE_PATH = DATA_DIR / "embed_cache.json"
_lock = threading.Lock()

_client: OpenAI | None = None
_cache: dict[str, list[float]] | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.embeddings_ready():
            raise RuntimeError(
                "Embedding endpoint not configured.\n"
                "Set EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL in .env, "
                "or use the lexical retriever instead."
            )
        _client = OpenAI(
            api_key=settings.embed_api_key,
            base_url=settings.embed_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    return _client


def _load_cache() -> dict[str, list[float]]:
    global _cache
    if _cache is None:
        if _CACHE_PATH.exists():
            try:
                _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _cache = {}
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")


def _key(text: str) -> str:
    h = hashlib.sha1(f"{settings.embed_model}\x00{text}".encode())
    return h.hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, using the on-disk cache where possible.

    Only the cache-miss texts are sent to the API; everything is then written
    back so subsequent runs hit the cache.
    """
    cache = _load_cache()
    keys = [_key(t) for t in texts]

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        client = _get_client()
        to_embed = [texts[i] for i in missing_idx]
        resp = client.embeddings.create(model=settings.embed_model, input=to_embed)
        for j, item in enumerate(resp.data):
            cache[keys[missing_idx[j]]] = list(item.embedding)
        with _lock:
            _save_cache()

    return [cache[k] for k in keys]


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
