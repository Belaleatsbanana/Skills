# Full-Duplex-Bench Evaluation

Full-Duplex-Bench is a benchmark for evaluating full-duplex spoken dialogue models on their turn-taking capabilities. It systematically assesses key conversational behaviors including pause handling, backchanneling, turn-taking, and interruption management.

## Overview

Full-Duplex-Bench evaluates full-duplex spoken dialogue systems (SDMs) that can listen and speak simultaneously, offering more natural interactions compared to traditional half-duplex models.

**Source:** https://github.com/DanielLin94144/Full-Duplex-Bench

## Subtests

The benchmark consists of four main evaluation dimensions:

1. **pause** - Evaluates model's ability to handle pauses in conversation
2. **backchannel** - Evaluates model's backchanneling behavior (e.g., 'uh-huh', 'yeah')
3. **turn_taking** - Evaluates model's turn-taking capabilities
4. **interruption** - Evaluates model's handling of interruptions

## Setup

### 1. Clone the Full-Duplex-Bench Repository

```bash
git clone https://github.com/DanielLin94144/Full-Duplex-Bench.git
cd Full-Duplex-Bench
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Dataset

You have two options for getting the dataset:

#### Option A: Automatic Download (Recommended)

Install the `gdown` package and let the script download automatically:

```bash
pip install gdown

python nemo_skills/dataset/fullduplexbench/prepare.py --download
```

This will:
- Automatically download the v1.0 dataset from Google Drive (~500MB)
- Extract it to a default location
- Process it for nemo-skills

You can specify a custom download location:
```bash
python nemo_skills/dataset/fullduplexbench/prepare.py \
  --download \
  --fdb_data_path /path/to/download/location
```

#### Option B: Manual Download

If automatic download doesn't work, download manually:

1. **Download Link**: [https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3)

2. Extract the dataset. The structure should look like:
```
Full-Duplex-Bench-data/
└── v1_0/
    ├── candor_pause_handling/
    │   ├── 1/
    │   │   ├── input.wav
    │   │   ├── pause.json
    │   │   └── transcription.json
    │   └── ...
    ├── candor_turn_taking/
    ├── icc_backchannel/
    ├── synthetic_pause_handling/
    └── synthetic_user_interruption/
```

3. Run the preparation script:
```bash
python nemo_skills/dataset/fullduplexbench/prepare.py \
  --fdb_data_path /path/to/Full-Duplex-Bench-data
```

Both options will:
- Convert the dataset to nemo-skills format
- Copy audio files to the nemo-skills data directory
- Create test.jsonl files for each subtest
- Generate __init__.py files with evaluation configurations

### 3. Configure Evaluation

Edit the configuration file (e.g., `scripts/fdb_s2s_offline_config.yaml`) to set:

- `fdb_repo_path`: Path to your Full-Duplex-Bench repository
- `data_dir`: Path where prepared dataset is stored
- `output_dir`: Where to save evaluation results
- `model`: Path to your model checkpoint
- Other cluster/server settings as needed

## Dependencies

For automatic dataset download:
```bash
pip install gdown
```

For audio processing:
```bash
pip install soundfile
```

## Usage

### Full Evaluation

Run all subtests with inference and scoring:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml
```

### Smoke Test

Quick test with limited samples:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_smoke_test.yaml
```

### Generation Only

Run only inference (skip scoring):

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml \
  --generation_only
```

### Scoring Only

Run only scoring (if generation already completed):

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml \
  --scoring_only
```

### Specific Subtests

Evaluate only specific subtests:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml \
  --subtests pause,turn_taking
```

## Configuration Options

Key configuration parameters in the YAML file:

- `cluster`: Cluster name (e.g., oci_iad)
- `partition`: Compute partition for jobs
- `cpu_partition`: CPU-only partition for scoring
- `model`: Path to model checkpoint
- `server_type`: Server type (e.g., vllm)
- `server_gpus`: Number of GPUs for inference
- `num_chunks`: Number of parallel inference chunks
- `subtests`: List of subtests to run (or "all")
- `max_samples`: Limit samples for testing (optional)
- `fdb_repo_path`: Path to Full-Duplex-Bench repository
- `data_dir`: Path to prepared dataset
- `output_dir`: Output directory for results

## Output Structure

After running evaluation, the output directory will contain:

```
output_dir/
├── eval-results/
│   ├── fullduplexbench.pause/
│   │   ├── output.jsonl              # Model predictions
│   │   ├── fdb_format.jsonl          # Converted to FDB format
│   │   ├── metrics.json              # Evaluation metrics
│   │   └── summarized-results/       # Scoring logs
│   ├── fullduplexbench.backchannel/
│   ├── fullduplexbench.turn_taking/
│   └── fullduplexbench.interruption/
└── generation_results/
    └── ... (inference artifacts)
```

## Metrics

Full-Duplex-Bench uses automatic metrics for consistent evaluation:

- **Takeover Rate**: Quantifies conversational control dynamics
- **Jensen-Shannon Divergence (JSD)**: Measures distributional differences
- Task-specific metrics depending on the subtest

## Scoring Setup

The scoring uses the original Full-Duplex-Bench evaluation scripts. The integration:

1. Converts nemo-skills output to FDB format
2. Calls the original FDB scoring scripts
3. Parses and saves metrics in nemo-skills format

## Troubleshooting

### Audio Files Not Found

Make sure to run `prepare.py` with the correct `--fdb_data_path` pointing to the Full-Duplex-Bench data directory.

### Scoring Errors

Check that:
- `fdb_repo_path` is correct in your config
- Full-Duplex-Bench dependencies are installed
- The FDB repository has the expected `evaluate.py` script

### Model Backend Issues

For S2S models, ensure:
- `extra_decoding_seconds` is set to 0 (Full-Duplex-Bench default)
- Correct config paths and checkpoints are specified
- Server container has necessary dependencies

## References

- **Paper**: [Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities](https://arxiv.org/abs/2503.04721)
- **Repository**: https://github.com/DanielLin94144/Full-Duplex-Bench
- **Related**: Full-Duplex-Bench v2, MTR-DuplexBench for extended evaluation
