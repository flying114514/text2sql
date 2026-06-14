#!/bin/sh
# Container entrypoint: prepare data, seed the Postgres governance demo, serve.
set -e

# Use the docker-specific connections file (points at the `postgres` service,
# not localhost). No secret in it — the readonly password comes from the env.
cp -f connections.docker.yaml connections.yaml

# 1. Local SQLite sample DB — always available, needs no network or LLM.
echo "[entrypoint] building sample SQLite database..."
uv run python scripts/make_sample_db.py

# 2. Seed the real Postgres demo + provision the read-only role (best-effort).
#    If Postgres is unreachable the app still runs against the SQLite sample.
if [ -n "$PG_ADMIN_URL" ]; then
  echo "[entrypoint] seeding Postgres governance demo..."
  uv run python scripts/seed_postgres.py \
      --admin-url "$PG_ADMIN_URL" \
      --readonly-password "${PG_PASSWORD:-ro_pw_2026}" \
    || echo "[entrypoint] WARN: Postgres seed failed; continuing with SQLite only."
fi

# 3. Serve on all interfaces so the published port is reachable from the host.
echo "[entrypoint] starting web server on 0.0.0.0:8000 ..."
exec uv run python scripts/serve.py --host 0.0.0.0 --port 8000
