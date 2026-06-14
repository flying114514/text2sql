"""Phase 0 smoke test #2 — verify LLM connectivity (requires a configured .env).

uv run python scripts/smoke_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.config import settings  # noqa: E402
from text2sql.llm import ping  # noqa: E402


def main() -> None:
    print(f"provider base_url : {settings.llm_base_url}")
    print(f"model             : {settings.llm_model}")
    print("calling LLM ...")
    reply = ping()
    print(f"reply             : {reply!r}")
    assert "pong" in reply.lower(), "unexpected reply from model"
    print("\nLLM CONNECTIVITY OK [OK]")


if __name__ == "__main__":
    main()
