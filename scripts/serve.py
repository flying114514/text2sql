"""Run the Text2SQL web app.

    uv run python scripts/serve.py            # http://127.0.0.1:8000
    uv run python scripts/serve.py --port 9000

Open the printed URL in a browser. The page talks to the FastAPI backend, which
runs the full agent (retrieval + few-shot + self-correction) with tracing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"\n  Text2SQL Agent  ->  http://{args.host}:{args.port}\n")
    uvicorn.run("text2sql.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
