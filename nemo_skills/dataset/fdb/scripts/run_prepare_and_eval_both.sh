#!/usr/bin/env bash
# Prepare FDB v1.0 and v1.5, then run s2s voicechat offline evaluation for both.
#
# IMPORTANT: Run this script ON THE CLUSTER (where your repo lives on lustre).
# The eval config uses data_dir on lustre; the prepared test.jsonl must exist at
# that path. If you only ran prepare locally, either run this script on the
# cluster or rsync the prepared fdb dir to the cluster at the same path.
#
# Run from Skills/ with PYTHONPATH=$(pwd). For v1.5: ensure Full-Duplex-Bench-data
# on the cluster contains v1.5/ or v1_5/, or install gdown to download.

set -e
SKILLS_DIR="/home/mmkrtchyan/projects/speechLM/s2s/Skills"
cd "$SKILLS_DIR"
export PYTHONPATH="${SKILLS_DIR}:${PYTHONPATH}"

# Paths: use same lustre paths as in fdb_s2s_offline_*_config.yaml so prepared data is where eval expects it
FDB_DATA="${FDB_DATA_PATH:-$SKILLS_DIR/../Full-Duplex-Bench-data}"
CONFIG_DIR="nemo_skills/dataset/fdb/scripts"

echo "=== 1) Prepare FDB v1.0 ==="
python -m nemo_skills.dataset.fdb.prepare --fdb_data_path "$FDB_DATA" --version v1.0

echo ""
echo "=== 2) Prepare FDB v1.5 (skips if v1.5 data missing; install gdown to download) ==="
python -m nemo_skills.dataset.fdb.prepare --fdb_data_path "$FDB_DATA" --version v1.5 || true

echo ""
echo "=== 3) Run s2s voicechat offline eval for v1.0 ==="
python nemo_skills/dataset/fdb/scripts/run_eval.py \
  --config "$CONFIG_DIR/fdb_s2s_offline_v1.0_config.yaml"

echo ""
echo "=== 4) Run s2s voicechat offline eval for v1.5 ==="
python nemo_skills/dataset/fdb/scripts/run_eval.py \
  --config "$CONFIG_DIR/fdb_s2s_offline_v1.5_config.yaml"

echo ""
echo "Done. Check output_dir in the configs for eval-results and metrics.json."
