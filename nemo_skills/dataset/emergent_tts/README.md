## EmergentTTS-Eval dataset (`emergent_tts`)

This dataset integration lets you:

- **Prepare** the EmergentTTS-Eval test set under a shared `data_dir` (download baseline audios + metadata + MOS model).
- **Generate** TTS outputs with NeMo-Skills (`ns eval` via `run_tts_eval.py`).
- **Score** the generated outputs with EmergentTTS-Eval (WER/MOS/win-rate, depending on config).

### 1) Prepare the test set (requires `HF_TOKEN`)

`prepare.py` downloads the dataset and writes all required artifacts into:

- `<DATA_DIR>/emergent_tts/emergent/test.jsonl`
- `<DATA_DIR>/emergent_tts/data/emergent_tts_eval_data.jsonl`
- `<DATA_DIR>/emergent_tts/data/baseline_audios/*.wav`
- `<DATA_DIR>/emergent_tts/data/wv_mos.ckpt`

Run it from your dev machine (or any environment with network access):

```bash
cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-tts-eval
. ./.venv/bin/activate

export HF_TOKEN="<your_hf_token>"

python nemo_skills/dataset/emergent_tts/prepare.py \
  --output_dir "<DATA_DIR>/emergent_tts"
```

Optional flags:

- `--num_samples 10`: write only the first 10 samples (smoke test).
- `--overwrite`: re-download / regenerate outputs.

### 2) Configure evaluation

Use the example configs in `nemo_skills/dataset/emergent_tts/scripts/config/`.

In `scripts/config/default.yaml`, set:

- `generation.data_dir: <DATA_DIR>`
- `scoring.emergent_data_dir: <DATA_DIR>/emergent_tts/data`
- `scoring.scoring_code_path: <PATH_TO>/EmergentTTS-Eval-public` (on the cluster)

### 3) Clone + patch EmergentTTS-Eval-public for NVIDIA Inference API judging

On EOS (or wherever you run scoring), clone EmergentTTS-Eval:

```bash
cd /lustre/fsw/llmservice_nemo_speechlm/users/vmendelev/code
git clone <repo_url> EmergentTTS-Eval-public
```

Then update Emergent’s judge client selection so that **Gemini models are called via NVIDIA’s OpenAI-compatible Inference API**.

Target behavior:

- **Model name** stays as: `gcp/google/gemini-2.5-pro` (or similar).
- **Base URL** is NVIDIA Inference API: `https://inference-api.nvidia.com/v1`
- **API key** comes from: `JUDGER_API_KEY` (or `NVIDIA_API_KEY`)

Minimal patch checklist inside `EmergentTTS-Eval-public`:

- In `api_clients.py` (or wherever the client is chosen), ensure `gcp/google/*` uses an **OpenAI-compatible** client (not the Google SDK client), e.g.:
  - `OpenAI(base_url=<judger_base_url>, api_key=os.getenv("JUDGER_API_KEY"))`
- Thread `judger_base_url` through so calls use `https://inference-api.nvidia.com/v1` (not the full `/v1/chat/completions` endpoint).

After patching, set these in `scripts/config/default.yaml`:

- `scoring.judge_model: gcp/google/gemini-2.5-pro`
- `scoring.judger_base_url: https://inference-api.nvidia.com/v1/chat/completions`

### 3) Run evaluation (generation + scoring)

From your dev machine, submit jobs to EOS:

```bash
cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-tts-eval
. ./.venv/bin/activate
mkdir -p .nemo_run

export NEMORUN_HOME="$PWD/.nemo_run"
export NEMO_SKILLS_CONFIG_DIR=/home/vmendelev/workspace/expressiveness/src/ns_eval/cluster_configs
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

# Required for win-rate judging (NVIDIA Inference API key)
export JUDGER_API_KEY="<your_nvidia_api_key>"

python -m nemo_skills.dataset.emergent_tts.scripts.run_tts_eval \
  --config nemo_skills/dataset/emergent_tts/scripts/config/default.yaml \
  --stage all \
  --expname emergent_eval
```

### 4) Smoke test (10 samples, interactive)

```bash
cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-tts-eval
. ./.venv/bin/activate
mkdir -p .nemo_run

export NEMORUN_HOME="$PWD/.nemo_run"
export NEMO_SKILLS_CONFIG_DIR=/home/vmendelev/workspace/expressiveness/src/ns_eval/cluster_configs
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

python -m nemo_skills.dataset.emergent_tts.scripts.run_tts_eval \
  --config nemo_skills/dataset/emergent_tts/scripts/config/interactive_10.yaml \
  --stage generation \
  --expname emergent_smoke10
```

### Outputs

NeMo-Skills generation writes:

- `<output_dir>/eval-results/emergent_tts.emergent/output.jsonl`
- `<output_dir>/eval-results/emergent_tts.emergent/audio/*.wav` (or equivalent)

Emergent scoring writes (in the same benchmark folder):

- `emergent-tts-eval_*_evaluation-predictions.jsonl`
- `emergent-tts-eval_*_evaluation-metrics.json`
- `metrics.json` (a NeMo-Skills-friendly copy of Emergent metrics)

