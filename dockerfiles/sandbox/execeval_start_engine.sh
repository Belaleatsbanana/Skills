#!/bin/bash
# ExecEval gunicorn entry (ntunlp/ExecEval execution_engine).
# NeMo pipeline sets NGINX_PORT and LISTEN_PORT for the sandbox sidecar; map them to Gunicorn's bind port.
export GUNICORN_PORT="${GUNICORN_PORT:-${NGINX_PORT:-${LISTEN_PORT:-5000}}}"
set -euo pipefail

for ((i = 0; i < NUM_WORKERS; i++)); do
  I_GID=$((RUN_GID + i))
  I_UID=$((RUN_UID + i))
  groupadd -g "${I_GID}" "runner${I_GID}" 2>/dev/null || true
  useradd -M "runner${I_UID}" -g "${I_GID}" -u "${I_UID}" 2>/dev/null || true
done

cd /root/execution_engine
exec gunicorn \
  -w "${NUM_WORKERS}" \
  --bind "0.0.0.0:${GUNICORN_PORT}" \
  --timeout 0 \
  --log-level "${LOG_LEVEL}" \
  "wsgi:app"
