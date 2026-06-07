#!/usr/bin/env bash
# Production startup script for EquityLens API
# Used by Render as the start command.
set -e

echo "Starting EquityLens API..."
echo "Port: ${PORT:-8000}"
echo "Debug: ${DEBUG:-false}"

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WEB_CONCURRENCY:-4}" \
    --log-level "${LOG_LEVEL:-info}" \
    --no-access-log
