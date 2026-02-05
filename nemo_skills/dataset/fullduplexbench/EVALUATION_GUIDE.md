# Full-Duplex-Bench Evaluation Guide

Comprehensive guide for running and analyzing Full-Duplex-Bench evaluations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Evaluation Workflow](#evaluation-workflow)
3. [Configuration Options](#configuration-options)
4. [Scoring and Metrics](#scoring-and-metrics)
5. [Result Analysis](#result-analysis)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Quick Start

### Minimal Setup

```bash
# 1. Download dataset
python nemo_skills/dataset/fullduplexbench/prepare.py --download

# 2. Configure evaluation
cp nemo_skills/dataset/fullduplexbench/scripts/fdb_eval_config.yaml my_config.yaml
# Edit my_config.yaml with your paths and settings

# 3. Run evaluation
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config my_config.yaml
```

### Smoke Test

```bash
# Quick test with 10 samples per subtest
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_smoke_test.yaml
```

## Evaluation Workflow

### 1. Dataset Preparation

**Option A: Automatic Download**

```bash
pip install gdown
python nemo_skills/dataset/fullduplexbench/prepare.py --download
```

**Option B: Manual Download**

```bash
# Download from: https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3
python nemo_skills/dataset/fullduplexbench/prepare.py \
  --fdb_data_path /path/to/downloaded/data
```

### 2. Configuration

Key settings in your config YAML:

```yaml
# Cluster and compute resources
cluster: oci_iad
partition: batch_block1,batch_block3
cpu_partition: cpu

# Model configuration
model: /path/to/model
server_type: vllm
server_gpus: 1

# Paths
data_dir: /path/to/fullduplexbench/data
output_dir: /path/to/output
fdb_repo_path: /path/to/Full-Duplex-Bench

# Evaluation settings
subtests: all  # or: pause, backchannel, turn_taking, interruption
max_samples: null  # null = all samples, or set to integer for testing
```

### 3. Running Evaluation

**Full Evaluation (Generation + Scoring)**

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config my_config.yaml
```

**Generation Only**

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config my_config.yaml \
  --generation_only
```

**Scoring Only** (after generation completed)

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config my_config.yaml \
  --scoring_only
```

**Specific Subtests**

```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config my_config.yaml \
  --subtests pause turn_taking
```

### 4. Alternative Evaluation Methods

**Using External API/Server**

For models served via API (not self-hosted):

```yaml
# In your config:
server_type: openai
server_gpus: 0  # External API
server_address: "http://your-api-endpoint/v1"
api_key_env_var: "YOUR_API_KEY"
model: "model-name"
server_server_type: "openai"
```

**Standalone Scoring**

For detailed scoring output and debugging:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/run_fdb_scoring_standalone.py \
  --eval_results_dir /path/to/eval-results/fullduplexbench.pause \
  --fdb_repo /path/to/Full-Duplex-Bench \
  --subtest pause \
  --force  # Re-run even if metrics exist
```

## Configuration Options

### Cluster Configuration

```yaml
cluster: oci_iad              # Cluster name
partition: batch_block1       # GPU partition for generation
cpu_partition: cpu            # CPU partition for scoring
```

### Model Configuration

```yaml
model: /path/to/model         # Model checkpoint path
server_type: vllm             # vllm, trt_llm, openai, etc.
server_gpus: 1                # Number of GPUs (0 for external API)
server_address: null          # External API address (optional)
server_container: null        # Container image (optional)
server_entrypoint: null       # Container entrypoint (optional)
server_args: ""               # Additional server arguments
```

### Evaluation Configuration

```yaml
subtests: all                 # "all" or list: [pause, turn_taking]
max_samples: null             # null = all, or integer for testing
num_chunks: 1                 # Parallel processing chunks
expname: fullduplexbench      # Experiment name
```

### Run Modes

```yaml
generation_only: false        # Only run inference
scoring_only: false           # Only run scoring
dry_run: false                # Print commands without executing
```

## Scoring and Metrics

### Full-Duplex-Bench Metrics

Each subtest produces specific metrics:

#### Common Metrics
- **Takeover Rate**: Measures conversational control dynamics
- **Jensen-Shannon Divergence (JSD)**: Distributional similarity to reference
- **Accuracy**: Task-specific accuracy
- **F1 Score**: Precision and recall balance

#### Subtest-Specific Metrics

**Pause Handling**
- Pause detection accuracy
- Pause duration alignment
- False positive/negative rates

**Backchannel**
- Backchannel timing accuracy
- Appropriateness score
- Type distribution (uh-huh, yeah, etc.)

**Turn-Taking**
- Turn prediction accuracy
- Overlap handling
- Turn transition smoothness

**Interruption**
- Interruption detection rate
- Response appropriateness
- Recovery capability

### Metrics Output Format

Results are saved in `eval-results/fullduplexbench.{subtest}/metrics.json`:

```json
{
  "fullduplexbench.pause": {
    "greedy": {
      "takeover_rate": 0.45,
      "jsd": 0.23,
      "accuracy": 0.82,
      "f1_score": 0.78
    }
  }
}
```

## Result Analysis

### Aggregate Results

Collect and summarize results across all subtests:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/aggregate_results.py \
  --eval_results_dir /path/to/eval-results \
  --csv aggregate_results.csv \
  --json summary.json
```

Output:
- Console summary with mean/min/max statistics
- CSV file with per-subtest metrics
- JSON file with complete summary

### Compare Multiple Runs

Compare different models or configurations:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/compare_eval_results.py \
  --runs \
    baseline:/path/to/baseline/eval-results \
    improved:/path/to/improved/eval-results \
  --baseline baseline \
  --output comparison.csv \
  --report comparison_report.md
```

Features:
- Side-by-side comparison
- Delta calculation vs baseline
- Percentage improvements
- Best performer highlighting
- Markdown report generation

### Filter Specific Metrics

Focus on particular metrics:

```bash
python nemo_skills/dataset/fullduplexbench/scripts/compare_eval_results.py \
  --runs model1:/path/1 model2:/path/2 \
  --metric takeover_rate
```

## Troubleshooting

### Common Issues

#### Dataset Not Found

```
Error: Full-Duplex-Bench data path not found
```

**Solution:**
```bash
# Re-run preparation
python nemo_skills/dataset/fullduplexbench/prepare.py --download
# Or specify correct path
python prepare.py --fdb_data_path /correct/path
```

#### Audio Files Missing

```
Warning: input.wav not found
```

**Solution:** Ensure dataset was properly extracted:
```bash
ls /path/to/Full-Duplex-Bench-data/v1.0/candor_pause_handling/
# Should show directories: 1/, 2/, 3/, etc.
```

#### Scoring Fails

```
Error: Could not find evaluation script
```

**Solution:** Verify FDB repository setup:
```bash
ls /path/to/Full-Duplex-Bench/
# Should contain: evaluate.py or scripts/evaluate.py
```

#### No Metrics Extracted

```
Warning: Could not extract any metrics from FDB output
```

**Solution:** Use standalone scorer for detailed output:
```bash
python scripts/run_fdb_scoring_standalone.py \
  --eval_results_dir /path/to/eval-results/fullduplexbench.pause \
  --fdb_repo /path/to/Full-Duplex-Bench \
  --subtest pause
```

### Debug Mode

Enable verbose output:

```bash
# Dry run to see commands
python generate_from_api_and_score_official.py \
  --config my_config.yaml \
  --dry_run

# Use standalone scorer for details
python run_fdb_scoring_standalone.py \
  --eval_results_dir /path \
  --fdb_repo /path \
  --subtest pause \
  --keep_converted  # Keep intermediate files
```

## Best Practices

### 1. Start with Smoke Test

Always run a smoke test first:

```bash
# Test with 10 samples
python generate_from_api_and_score_official.py \
  --config fdb_smoke_test.yaml
```

### 2. Separate Generation and Scoring

For large evaluations, run separately:

```bash
# Generation (GPU-intensive)
python generate_from_api_and_score_official.py \
  --config my_config.yaml \
  --generation_only

# Scoring (CPU-only, cheaper)
python generate_from_api_and_score_official.py \
  --config my_config.yaml \
  --scoring_only
```

### 3. Use Parallel Processing

Speed up generation with chunks:

```yaml
# In config
num_chunks: 10  # Process 10 chunks in parallel
```

### 4. Track Experiments

Organize output directories:

```bash
output_dir: /experiments/fullduplexbench/model-v1.0-20250205
output_dir: /experiments/fullduplexbench/model-v2.0-20250210
```

### 5. Document Configurations

Keep config files with results:

```bash
# Copy config to output directory
cp my_config.yaml /path/to/output/config.yaml
```

### 6. Regular Backups

Backup important results:

```bash
# Backup metrics
tar -czf fdb-results-$(date +%Y%m%d).tar.gz eval-results/
```

### 7. Compare Incrementally

Track improvements over time:

```bash
python compare_eval_results.py \
  --runs \
    v1.0:/experiments/v1.0/eval-results \
    v1.5:/experiments/v1.5/eval-results \
    v2.0:/experiments/v2.0/eval-results \
  --baseline v1.0 \
  --report progress_report.md
```

## Advanced Usage

### Custom Metrics Parsing

Modify `run_fdb_scoring.py` to extract custom metrics:

```python
# Add to parse_metrics() function
metric_patterns = {
    'custom_metric': r'custom[:\s]+([0-9.]+)',
}
```

### Batch Evaluation

Evaluate multiple models:

```bash
for model in model1 model2 model3; do
  python generate_from_api_and_score_official.py \
    --config base_config.yaml \
    --model /path/to/$model \
    --output_dir /results/$model
done
```

### Integration with CI/CD

Automate evaluation in pipelines:

```yaml
# .gitlab-ci.yml
evaluate:
  script:
    - python prepare.py --download
    - python generate_from_api_and_score_official.py --config ci_config.yaml
    - python aggregate_results.py --eval_results_dir eval-results --json results.json
  artifacts:
    paths:
      - results.json
```

## References

- [Full-Duplex-Bench Paper](https://arxiv.org/abs/2503.04721)
- [Full-Duplex-Bench Repository](https://github.com/DanielLin94144/Full-Duplex-Bench)
- [README.md](./README.md) - Basic setup guide
- [INTEGRATION_SUMMARY.md](./INTEGRATION_SUMMARY.md) - Technical integration details
