"""Central configuration, loaded from environment / .env file.

We deliberately use ONE OpenAI-compatible client for every provider, so the
only thing that changes between DeepSeek / Qwen / Zhipu / OpenAI is three env
vars: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/text2sql/config.py -> repo root)
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Reads from a .env file at the project root (and real environment vars,
    which take precedence). Missing required values raise a clear error at
    startup instead of failing deep inside an API call.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (OpenAI-compatible) ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # --- Generation behaviour ---
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.0

    # --- Answer cache (Phase 11) ---
    # Skip the whole pipeline when the same first-turn question returns. The
    # key includes the governance role (different roles see different data) and
    # entries expire after the TTL so a changed database is not served stale.
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 24h

    # --- Feedback flywheel (Phase 11b) ---
    # 👍 turns a (question -> SQL) pair into a verified, same-database few-shot
    # example; 👎 logs the miss and invalidates that question's cache. Up to
    # feedback_fewshot_k verified examples are injected ahead of the Spider pool.
    feedback_enabled: bool = True
    feedback_fewshot_k: int = 3

    # --- Pinned questions / dashboard (Phase 11c) ---
    # Pin a question (not its answer) into a re-runnable dashboard card, bound to
    # the db + governance role that produced it.
    pins_enabled: bool = True

    # --- Reliability (Phase 5) ---
    # SQL query wall-clock timeout (seconds); a watchdog interrupts the query.
    db_query_timeout: float = 5.0
    # Optional fallback model (used if the primary call fails). If only
    # fallback_model is set, the primary base_url/api_key are reused; otherwise
    # a fully separate provider can be configured.
    fallback_model: str = ""
    fallback_base_url: str = ""
    fallback_api_key: str = ""

    # --- Observability (Phase 5) ---
    # Cost estimate pricing (USD per 1M tokens) — shared by tracing and eval.
    price_in_per_m: float = 0.27
    price_out_per_m: float = 1.10
    # Optional Langfuse cloud backend; local JSONL tracing is always on.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Embeddings (optional, for the semantic schema retriever) ---
    # Any OpenAI-compatible embedding endpoint. Leave empty to fall back to the
    # zero-dependency lexical retriever. Can reuse the same provider as the LLM
    # (e.g. SiliconFlow / DashScope / OpenAI) by pointing these at it.
    embed_api_key: str = ""
    embed_base_url: str = ""
    embed_model: str = ""

    def embeddings_ready(self) -> bool:
        """True only if a full embedding endpoint is configured."""
        return bool(self.embed_api_key and self.embed_base_url and self.embed_model)

    def fallback_ready(self) -> bool:
        """True if a fallback model is configured (provider may be reused)."""
        return bool(self.fallback_model)

    def langfuse_ready(self) -> bool:
        """True if Langfuse cloud credentials are configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def gateway_ready(self) -> bool:
        """True if a gateway.yaml exists (the LLM gateway is configured).

        Lives here so callers depend on settings, not on the gateway package —
        but the authoritative check (does it actually have providers?) is
        gateway.providers.gateway_enabled().
        """
        return (ROOT_DIR / "gateway.yaml").exists()

    def assert_ready(self) -> None:
        """Fail fast with a friendly message if the API key is not set."""
        if not self.llm_api_key or self.llm_api_key.startswith("sk-xxxx"):
            raise RuntimeError(
                "LLM_API_KEY is not configured.\n"
                "Copy .env.example to .env and fill in your real key, base_url and model."
            )


# A single shared instance imported everywhere.
settings = Settings()
