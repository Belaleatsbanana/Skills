# TTS Evaluation Based on NeMo-Skills

This is an adaptation of `examples/tts/magpietts_inference.py` into NeMo-Skills. The generation and scoring are separated into 2 stages and can be effectively parallelized. The same code as in `magpietts_inference.py` is used for both stages.

The test sets are also borrowed from the current evaluation setup.

## Getting Started

### 1. Clone and Setup

```bash
# Clone this branch
git clone <repository_url>
cd ns_eval

# Create a virtual environment and install nemo-skills
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Cluster Configuration

Decide which cluster you want to work on and setup the corresponding cluster configuration.

- An example configuration for EOS is provided in `cluster_configs/eos_example.yaml`
- You can get more configurations from the [NeMo-Skills cluster configs](https://github.com/NVIDIA/NeMo-Skills/tree/main/cluster_configs)
- Update the username in the configuration file

Note that NeMo-Skills standard quant of resources is 1 gpu. Eos cluster is special because it allows assigning only full nodes (not e.g. 2 gpus from 8). So I had to write a fix, which is not nice, so probably won't be merged as is. You can remove the "EOS FIX 8 chunks per node" if running on other clusters. This may entail small changes in the config.

### 3. Prepare Test Data

You can either prepare a new test set or reuse an existing data directory.

**To prepare a new test set:**

```bash
cd /home/vmendelev/workspace/expressiveness/src/ns_eval && source .venv/bin/activate && \
python nemo_skills/dataset/nv_tts/prepare.py \
  --config <cluster_login>:<path_to_>/eval_config_full_fixed.json
```

This will prepare `test.jsonl` for each benchmark with pointers to the files on the cluster.

**To reuse an existing data directory (EOS):**

```
/lustre/fsw/llmservice_nemo_speechlm/users/vmendelev/tmp/data_dir
```

### 4. Configuration Files

Review the config file and ensure all required artifacts are in the specified locations:

| Config | Description |
|--------|-------------|
| `nemo_skills/dataset/nv_tts/scripts/config/default.yaml` | For `.nemo` model input |
| `nemo_skills/dataset/nv_tts/scripts/config/grpo_small_step1100.yaml` | For checkpoint + hparams input |

### 5. Environment Setup

Make sure `HF_TOKEN` is present in the environment:

```bash
export HF_TOKEN=<your_huggingface_token>
# or source from your .env file
. ~/.env && export HF_TOKEN=$HF_READ_ONLY
```

## Running Evaluation

### Full Evaluation (Generation + Scoring)

```bash
cd /home/vmendelev/workspace/expressiveness/src/ns_eval && source .venv/bin/activate && \
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
python -m nemo_skills.dataset.nv_tts.scripts.run_tts_eval \
  --config nemo_skills/dataset/nv_tts/scripts/config/default.yaml \
  --stage all \
  --expname default_eval
```

### Stage Options

| Stage | Description |
|-------|-------------|
| `all` | Run both generation and scoring |
| `generation` | Run only TTS generation |
| `scoring` | Run only scoring (requires completed generation) |
| `aggregation` | Print summary of all metrics |

## Comparing Results

To produce a comparison report between different evaluation runs:

```bash
cd /home/vmendelev/workspace/expressiveness/src/ns_eval && source .venv/bin/activate && \
python nemo_skills/dataset/nv_tts/scripts/compare_eval_results.py \
  --baseline_dir <path_to_baseline_results> \
  --compare_dir <path_to_comparison_results> \
  --output_file tts_comparison_report.md
```

See [example report](scripts/tts_comparison_report.md) for sample output.

## Output Structure

Results are saved to `output_dir/eval-results/` with the following structure:

```
output_dir/
├── eval-results/
│   ├── nv_tts.libritts_seen/
│   │   ├── output.jsonl          # Generated audio paths + metadata
│   │   ├── output_with_metrics.jsonl  # With per-file metrics
│   │   ├── metrics.json          # Aggregate metrics (CER, WER, UTMOSv2)
│   │   └── audio/                # Generated audio files
│   ├── nv_tts.vctk/
│   │   └── ...
│   └── ...
└── eval-logs/                    # Job logs
```
