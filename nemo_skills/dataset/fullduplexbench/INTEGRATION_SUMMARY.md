# Full-Duplex-Bench Integration Summary

This document summarizes the integration of Full-Duplex-Bench 1.0 benchmark into the nemo-skills repository, following the VoiceBench pattern.

## Important Note on Data Acquisition

**Unlike VoiceBench**, Full-Duplex-Bench does not host its dataset on HuggingFace. However, we've added **automatic download support**:

- **Automatic Download**: Use `python prepare.py --download` (requires `gdown` package)
  - Downloads v1.0 only (~500MB)
  - v1.0 folder ID: `1hxzRk7xgtdr5ZEoctnp0sFK0COv91W3h`
- **Manual Download**: https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3
  - Download only the v1.0 folder
- **Dataset Version**: Currently supports v1.0 (v1.5 support can be added later)

The automatic download feature uses the `gdown` library to fetch the dataset directly from Google Drive, making the setup process as easy as VoiceBench.

## Created Files

### Core Files

1. **`__init__.py`**
   - Defines the benchmark group and subtests
   - Four subtests: pause, backchannel, turn_taking, interruption
   - Group: `speechlm`

2. **`prepare.py`**
   - Dataset preparation script
   - Converts Full-Duplex-Bench data to nemo-skills format
   - Handles audio file extraction and storage
   - Creates three message variants: audio-only, text+audio, text-only

3. **`README.md`**
   - Comprehensive documentation
   - Setup instructions
   - Usage examples
   - Configuration options
   - Troubleshooting guide

### Scripts Directory

4. **`scripts/generate_from_api_and_score_official.py`**
   - Main launch script that chains inference and scoring
   - Supports all configuration options (dry_run, generation_only, scoring_only)
   - CLI overrides for flexibility
   - Based on VoiceBench pattern

5. **`scripts/run_fdb_scoring.py`**
   - Scoring wrapper for Full-Duplex-Bench official evaluation
   - Converts nemo-skills output to FDB format
   - Calls original FDB scoring scripts
   - Parses metrics and saves in nemo-skills format

6. **`scripts/convert_to_fdb_format.py`**
   - Format converter from nemo-skills to Full-Duplex-Bench format
   - Cleans special tokens (timing markers) from S2S model output
   - Preserves metadata fields

### Configuration Files

7. **`scripts/fdb_s2s_offline_config.yaml`**
   - Full evaluation configuration for S2S offline backend
   - Production-ready settings
   - All four subtests

8. **`scripts/fdb_eval_config.yaml`**
   - Basic evaluation configuration template
   - Easy to customize for different setups

9. **`scripts/fdb_smoke_test.yaml`**
   - Quick test configuration with limited samples
   - Single subtest (pause)
   - 10 samples max for fast iteration

## Architecture

The integration follows the same pattern as VoiceBench:

```
fullduplexbench/
├── __init__.py                      # Benchmark definitions
├── prepare.py                       # Dataset preparation
├── README.md                        # Documentation
└── scripts/
    ├── generate_from_api_and_score_official.py   # Main launcher
    ├── run_fdb_scoring.py                        # Scoring wrapper
    ├── convert_to_fdb_format.py                  # Format converter
    ├── fdb_s2s_offline_config.yaml               # Full config
    ├── fdb_eval_config.yaml                      # Basic config
    └── fdb_smoke_test.yaml                       # Smoke test config
```

## Key Features

### 1. Dual-Phase Evaluation
- **Generation Phase**: Runs inference using nemo-skills pipeline
- **Scoring Phase**: Calls original Full-Duplex-Bench evaluation scripts

### 2. Flexible Configuration
- YAML-based configuration
- CLI overrides for key parameters
- Support for different server backends (vllm, megatron, etc.)
- Configurable parallelism with `num_chunks`

### 3. Official Scoring Integration
- Calls original FDB `evaluate.py` script
- Preserves exact metrics from FDB benchmark
- Converts output to nemo-skills format for consistency

### 4. Multiple Evaluation Modes
- Full evaluation (all subtests)
- Individual subtest evaluation
- Smoke testing with limited samples
- Generation-only or scoring-only modes

### 5. S2S Model Support
- Special token cleaning (timing markers)
- Audio file handling
- Proper `extra_decoding_seconds` settings (0 for FDB vs 20 for VoiceBench)

## Usage Workflow

### Step 1: Clone Full-Duplex-Bench Repository
```bash
git clone https://github.com/DanielLin94144/Full-Duplex-Bench.git
cd Full-Duplex-Bench
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

**Option A - Automatic (Recommended)**:
```bash
pip install gdown
python nemo_skills/dataset/fullduplexbench/prepare.py --download
```

**Option B - Manual**:
```bash
# Download from Google Drive manually, then:
python nemo_skills/dataset/fullduplexbench/prepare.py \
  --fdb_data_path /path/to/Full-Duplex-Bench-data
```

### Step 3: Configure Evaluation
Edit `fdb_s2s_offline_config.yaml`:
- Set `fdb_repo_path`
- Set `data_dir`
- Configure model and server settings

### Step 4: Run Evaluation
```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml
```

## Comparison with VoiceBench

| Aspect | VoiceBench | Full-Duplex-Bench |
|--------|------------|-------------------|
| Subtests | 12 | 4 |
| Data Source | **HuggingFace** (auto-download) | **Google Drive** (manual download) |
| Data Format | Test questions + audio | User speech input for turn-taking tests |
| extra_decoding_seconds | 20 | 0 |
| GPT Judge | Required for some subtests | Not used |
| Metrics | Task-specific | Takeover Rate, JSD, Latency |
| Focus | General speech QA | Turn-taking & interaction behavior |
| Dataset Size | ~5000+ samples | ~727 samples (v1.0) |

## Integration Points

### Documentation
- Added section to `docs/evaluation/speech-audio.md`
- Explains setup, usage, and metrics
- Links to detailed README

### Code Pattern
- Follows VoiceBench structure exactly
- Same script naming conventions
- Same config file patterns
- Same output directory structure

## Testing

### Smoke Test
Quick validation with limited samples:
```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_smoke_test.yaml
```

### Dry Run
Test without executing:
```bash
python nemo_skills/dataset/fullduplexbench/scripts/generate_from_api_and_score_official.py \
  --config nemo_skills/dataset/fullduplexbench/scripts/fdb_s2s_offline_config.yaml \
  --dry_run
```

## Output Structure

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

## Future Enhancements

Potential improvements:
1. Add support for Full-Duplex-Bench v1.5 (overlap handling)
2. Add support for Full-Duplex-Bench v2 (multi-turn with automated examiner)
3. Integration tests in `tests/` directory
4. Automated FDB repository setup
5. Support for custom metrics parsing

## References

- **Paper**: [Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities](https://arxiv.org/abs/2503.04721)
- **Repository**: https://github.com/DanielLin94144/Full-Duplex-Bench
- **VoiceBench**: Used as reference for integration pattern
- **nemo-skills**: Main repository for evaluation framework

## Notes

- The integration assumes FDB's `evaluate.py` script follows a standard pattern
- Metric parsing may need adjustment based on actual FDB output format
- The prepare.py script expects specific data formats from FDB repository
- Audio handling assumes standard WAV format at 16kHz sampling rate
