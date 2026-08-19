#!/usr/bin/env bash
# Production startup script for EquityLens API on Render.
set -e

echo "============================================"
echo "  EquityLens API  –  Production Startup"
echo "============================================"
echo "Port:       ${PORT:-8000}"
echo "Debug:      ${DEBUG:-false}"
echo "Workers:    ${WEB_CONCURRENCY:-4}"
echo "Python:     $(python --version 2>&1)"
echo "============================================"

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WEB_CONCURRENCY:-4}" \
    --log-level "${LOG_LEVEL:-info}" \
    --forwarded-allow-ips '*' \
    --proxy-headers \
    --no-access-log
