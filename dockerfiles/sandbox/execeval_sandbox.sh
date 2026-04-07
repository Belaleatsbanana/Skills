#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Build / run / squashfs export for the ExecEval-based sandbox image used with NeMo Skills eval.
#
# The image exposes the same entry as the default sandbox (/start-with-nginx.sh) so Slurm/local
# pipeline steps work with:
#   ns eval ... --sandbox_container /path/to/nemo-skills-sandbox-exec-eval.sqsh
#   ns eval ... --sandbox_container docker://nemo-skills-sandbox-exec-eval
#
# HTTP:
#   GET  /health                     — readiness (NeMo wait_for_sandbox)
#   POST /api/execute_code           — ExecEval API (not NeMo POST /execute)
#
# Usage (from NeMo-Skills repo root):
#   ./dockerfiles/sandbox/execeval_sandbox.sh build [image_tag]
#   ./dockerfiles/sandbox/execeval_sandbox.sh run   [image_tag]
#   ./dockerfiles/sandbox/execeval_sandbox.sh sqsh  [image_tag] [output.sqsh]
#
# Environment (run/sqsh): NUM_WORKERS, GUNICORN_PORT, RUN_UID, RUN_GID, LOG_LEVEL, EXECEVAL_CONTAINER_NAME

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKERFILE="${REPO_ROOT}/dockerfiles/Dockerfile.sandbox.execeval"

DEFAULT_TAG="nemo-skills-sandbox-exec-eval"
DEFAULT_SQSH="nemo-skills-sandbox-exec-eval.sqsh"

cmd="${1:-help}"
IMAGE_TAG="${2:-${DEFAULT_TAG}}"
SQSH_OUT="${3:-${REPO_ROOT}/${DEFAULT_SQSH}}"

cd "${REPO_ROOT}"

build_image() {
  local tag="$1"
  echo "Building ${tag} from ${DOCKERFILE} (platform linux/amd64)..."
  docker build \
    --platform=linux/amd64 \
    --tag="${tag}" \
    -f "${DOCKERFILE}" \
    .
}

case "${cmd}" in
  build)
    build_image "${IMAGE_TAG}"
    echo "Done. Image: ${IMAGE_TAG}"
    echo "Pass to eval, e.g.:  --sandbox_container ${IMAGE_TAG}"
    ;;
  run)
    if ! docker image inspect "${IMAGE_TAG}" &>/dev/null; then
      build_image "${IMAGE_TAG}"
    fi
    echo "Running ${IMAGE_TAG} (--network=host). Port from NGINX_PORT/LISTEN_PORT or GUNICORN_PORT (default 5000)."
    docker run --network=host --rm \
      --name="${EXECEVAL_CONTAINER_NAME:-local-execeval-sandbox}" \
      -e NUM_WORKERS="${NUM_WORKERS:-16}" \
      -e GUNICORN_PORT="${GUNICORN_PORT:-5000}" \
      -e NGINX_PORT="${NGINX_PORT:-}" \
      -e LISTEN_PORT="${LISTEN_PORT:-}" \
      -e LOG_LEVEL="${LOG_LEVEL:-info}" \
      -e RUN_UID="${RUN_UID:-1586}" \
      -e RUN_GID="${RUN_GID:-1586}" \
      "${IMAGE_TAG}"
    ;;
  sqsh)
    build_image "${IMAGE_TAG}"
    if ! command -v enroot &>/dev/null; then
      echo "enroot not found. Install enroot, then run:" >&2
      echo "  enroot import -o ${SQSH_OUT} -- dockerd://${IMAGE_TAG}" >&2
      exit 1
    fi
    echo "Importing docker image ${IMAGE_TAG} -> ${SQSH_OUT}"
    enroot import -o "${SQSH_OUT}" -- "dockerd://${IMAGE_TAG}"
    echo "Done. Use with eval, e.g.:  --sandbox_container ${SQSH_OUT}"
    ;;
  help | -h | --help)
    sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "Unknown command: ${cmd}. Use: build | run | sqsh | help" >&2
    exit 1
    ;;
esac
