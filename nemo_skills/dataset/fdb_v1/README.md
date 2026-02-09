# Full-Duplex-Bench Evaluation

Benchmark for full-duplex spoken dialogue models (pause, backchannel, turn-taking, interruption).
**Source:** https://github.com/DanielLin94144/Full-Duplex-Bench

## 1. Prepare data (once)

From **Skills** (or with `PYTHONPATH` including Skills):

```bash
cd /path/to/s2s/Skills
export PYTHONPATH="$(pwd):$PYTHONPATH"

# v1.0: download to s2s/Full-Duplex-Bench-data and build test.jsonl (fdb_v1/*)
pip install gdown soundfile
python -m nemo_skills.dataset.fdb_v1.prepare

# v1.5: prepare from same dataset root (must contain v1.5 or v1_5 folder); writes to fdb_v1_5/*
python -m nemo_skills.dataset.fdb_v1.prepare --version v1.5 --fdb_data_path /path/to/Full-Duplex-Bench-data
```

To use existing data (v1.0): `python -m nemo_skills.dataset.fdb_v1.prepare --fdb_data_path /path/to/Full-Duplex-Bench-data`
Manual download: [Google Drive](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3) → extract **v1.0** and/or **v1.5** subfolders.

## 2. Run evaluation

Edit `scripts/fdb_s2s_offline_config.yaml` (v1.0) or `scripts/fdb_s2s_offline_v1.5_config.yaml` (v1.5): set `data_dir`, `output_dir`, `fdb_repo_path`, `model`. Set `fdb_data_path` to the FDB dataset root (needed for **turn_taking** and **interruption**). For **interruption** (and **behavior**), set `NVIDIA_API_KEY` in the cluster config `env_vars`.

From **Skills**:

```bash
cd /path/to/s2s/Skills
export PYTHONPATH="$(pwd):$PYTHONPATH"

# v1.0 (fdb_v1.*)
python nemo_skills/dataset/fdb_v1/scripts/run_eval.py \
  --config nemo_skills/dataset/fdb_v1/scripts/fdb_s2s_offline_config.yaml

# v1.5 (fdb_v1_5.*) — separate config and output, run independently
python nemo_skills/dataset/fdb_v1/scripts/run_eval.py \
  --config nemo_skills/dataset/fdb_v1/scripts/fdb_s2s_offline_v1.5_config.yaml
```

**Options:** `--generation_only`, `--scoring_only`, `--scoring_force`, `--subtests pause,turn_taking`, `--max_samples 5`, `--dry_run`.
For scoring-only, pass `--output_dir` to an existing run that already has `eval-results/`.

## Output

- v1.0: `output_dir/eval-results/fdb_v1.{subtest}/` (from `fdb_s2s_offline_config.yaml`).
- v1.5: `output_dir/eval-results/fdb_v1_5.{subtest}/` (subtests: background_speech, talking_to_other, backchannel, interruption).

Each contains `output.jsonl`, `fdb_prepared/`, and `metrics.json`.

## Backchannel with S2S incremental

The **backchannel** subtest expects short listener reactions; S2S offline often produces full turns. To run backchannel with the **S2S incremental** backend instead:

```bash
python nemo_skills/dataset/fdb_v1/scripts/run_eval.py \
  --config nemo_skills/dataset/fdb_v1/scripts/fdb_s2s_incremental_config.yaml \
  --subtests backchannel
```

Edit `scripts/fdb_s2s_incremental_config.yaml` to set `model`, `server_args` (e.g. `--config_path`), `data_dir`, `output_dir`, `fdb_repo_path`, `fdb_data_path`, and `session_artifacts_dir` to your paths.

## Troubleshooting

- No audio / missing data → run `prepare` (or set `--fdb_data_path`).
- Scoring fails → check `fdb_repo_path` and that the FDB repo has `evaluation/evaluate.py` and `get_transcript/asr.py`.
- **Backchannel "More than one dimension in audio"** → FDB’s Silero VAD expects mono. The prepare step now converts stereo to mono when writing `fdb_prepared/.../output.wav`. Ensure `soundfile` is installed (included in `scoring_installation_command`).

## Reference

[Paper](https://arxiv.org/abs/2503.04721) · [Repository](https://github.com/DanielLin94144/Full-Duplex-Bench)
