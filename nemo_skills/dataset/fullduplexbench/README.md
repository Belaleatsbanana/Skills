# Full-Duplex-Bench Evaluation

Benchmark for full-duplex spoken dialogue models (pause, backchannel, turn-taking, interruption).
**Source:** https://github.com/DanielLin94144/Full-Duplex-Bench

## 1. Prepare data (once)

From **Skills** (or with `PYTHONPATH` including Skills):

```bash
cd /path/to/s2s/Skills
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Download v1.0 to s2s/Full-Duplex-Bench-data and build test.jsonl for all subtests
pip install gdown soundfile
python -m nemo_skills.dataset.fullduplexbench.prepare
```

To use existing data: `python -m nemo_skills.dataset.fullduplexbench.prepare --fdb_data_path /path/to/Full-Duplex-Bench-data`
Manual download: [Google Drive v1.0](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3) → extract only the **v1.0** subfolder.

## 2. Run evaluation

Edit `scripts/fdb_s2s_offline_config.yaml`: set `data_dir`, `output_dir`, `fdb_repo_path`, `model`. Set `fdb_data_path` to the FDB dataset root (needed for **turn_taking** and **interruption**, so `turn_taking.json` / `interrupt.json` are copied). For **interruption**, FDB’s evaluator may call the OpenAI API — set a valid `OPENAI_API_KEY` (or the env var FDB uses) to avoid 401 errors.

From **Skills**:

```bash
cd /path/to/s2s/Skills
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Full run (generation + scoring)
python nemo_skills/dataset/fullduplexbench/scripts/run_eval.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml
```

**Options:** `--generation_only`, `--scoring_only`, `--scoring_force`, `--subtests pause,turn_taking`, `--max_samples 5`, `--dry_run`.
For scoring-only, pass `--output_dir` to an existing run that already has `eval-results/`.

## Output

`output_dir/eval-results/fullduplexbench.{subtest}/` contains `output.jsonl`, `fdb_prepared/`, and `metrics.json` per subtest.

## Backchannel with S2S incremental

The **backchannel** subtest expects short listener reactions; S2S offline often produces full turns. To run backchannel with the **S2S incremental** backend instead:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/run_eval.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_incremental_config.yaml \
  --subtests backchannel
```

Edit `scripts/fdb_s2s_incremental_config.yaml` to set `model`, `server_args` (e.g. `--config_path`), `data_dir`, `output_dir`, `fdb_repo_path`, `fdb_data_path`, and `session_artifacts_dir` to your paths.

## Troubleshooting

- No audio / missing data → run `prepare` (or set `--fdb_data_path`).
- Scoring fails → check `fdb_repo_path` and that the FDB repo has `evaluation/evaluate.py` and `get_transcript/asr.py`.
- **Backchannel "More than one dimension in audio"** → FDB’s Silero VAD expects mono. The prepare step now converts stereo to mono when writing `fdb_prepared/.../output.wav`. Ensure `soundfile` is installed (included in `scoring_installation_command`).

## Reference

[Paper](https://arxiv.org/abs/2503.04721) · [Repository](https://github.com/DanielLin94144/Full-Duplex-Bench)
