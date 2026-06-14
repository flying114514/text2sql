# Text2SQL Agent — container image.
# Uses the official uv image (uv + Python preinstalled) and the standard
# layer-cached dependency install: deps first, then the app source.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# 1. Install dependencies first so this layer is cached across code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. App source, then build/install the project itself.
COPY . .
RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1

EXPOSE 8000

RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["./docker/entrypoint.sh"]
