#!/bin/bash
# NeMo pipeline always launches the sandbox with /start-with-nginx.sh (see nemo_skills/pipeline/utils/exp.py).
# This image runs ExecEval instead of uwsgi/nginx; delegate to start_engine.sh in execution_engine.
set -euo pipefail
exec bash /root/execution_engine/start_engine.sh
