#!/bin/bash
set -x
STAGE="${1:-all}"
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
python -m nemo_skills.dataset.nv_tts.scripts.run_tts_eval \
  --config nemo_skills/dataset/nv_tts/scripts/config/tts-roy.yaml \
  --stage "$STAGE" \
  --expname default_eval