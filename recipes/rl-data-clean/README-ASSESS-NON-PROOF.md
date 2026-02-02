# Non-Proof Problem Assessment Guide

Quick guide for assessing non-proof math problems with solutions for RL training.

## Overview

Two pipeline versions available:
- **120b version**: Uses `gpt-oss-120b` model
- **DeepSeek version**: Uses `DeepSeek-V3.2` model

Both assess problem-answer-solution triplets and output binary ACCEPT/REJECT decisions.

---

## 📋 Quick Start

### Step 1: Prepare Your Data

Your input JSONL must contain:
```json
{
  "problem": "...",
  "expected_answer": "...",
  "extracted_solution": "..."
}
```

### Step 2: Update Configuration

**For gpt-oss-120b:**
```bash
vim recipes/rl-data-clean/configs/assess-non-proof-solutions.yaml
# Update: input_file path
```

**For DeepSeek:**
```bash
vim recipes/rl-data-clean/configs/assess-non-proof-deepseek.yaml
# Update: input_file path
```

### Step 3: Run Pipeline

**For gpt-oss-120b:**
```bash
cd /home/wedu/NeMo-Skills
python recipes/rl-data-clean/pipeline/assess_non_proof_pipeline.py \
    --config assess-non-proof-solutions
```

**For DeepSeek:**
```bash
cd /home/wedu/NeMo-Skills
python recipes/rl-data-clean/pipeline/assess_non_proof_pipeline-deepseek.py \
    --config assess-non-proof-deepseek
```

---

## 🔍 Key Differences

| Feature | gpt-oss-120b | DeepSeek-V3.2 |
|---------|--------------|---------------|
| **Server Type** | vllm | sglang |
| **Server Nodes** | 1 | 2 |
| **Top P** | 1.0 | 0.95 |
| **Temperature** | 1.0 | 1.0 |
| **Tokens to Generate** | 120000 | 120000 |
| **Special Args** | `reasoning_effort=high`<br>`enable_soft_fail=True` | `thinking=true`<br>`endpoint_type=chat`<br>`max_concurrent_requests=1024` |
| **Server Args** | N/A | `--ep-size 16 --dp 16`<br>`--enable-dp-attention`<br>`--tool-call-parser deepseekv32`<br>`--reasoning-parser deepseek-v3`<br>`--mem-fraction-static=0.8` |

---

## 📂 Output Structure

Both pipelines produce:

```
/workspace/rl-data-clean/non-proof-clean/<model_name>/step-1-assess-complete-solution/
├── output.jsonl              # Raw LLM outputs
├── accepted.jsonl            # Problems that PASSED quality assessment
├── rejected.jsonl            # Problems that FAILED quality assessment
└── generation-logs/          # Detailed generation logs
```

---

## 🎯 What Gets Assessed?

The pipeline evaluates:

1. **Problem Quality**: Clarity, completeness, well-defined
2. **Solution Presence & Clarity**: Is solution present and understandable?
3. **Correctness**: Does solution lead to expected answer?
4. **Completeness**: Are all steps shown?
5. **Consistency**: Problem-answer-solution alignment

**Decision**: ACCEPT or REJECT based on overall suitability for RL training.

---

## 📊 Understanding Results

**Accepted samples** (`accepted.jsonl`):
- High-quality triplets ready for RL training
- All criteria met

**Rejected samples** (`rejected.jsonl`):
- Issues detected in problem, solution, or consistency
- Review `reasoning_content` field in `output.jsonl` to understand why

---

## 🛠 Customization

### Change Model Resources

Edit the config file:
```yaml
generate_kwargs:
  server_gpus: 8        # Number of GPUs per node
  server_nodes: 2       # Number of nodes
  num_chunks: 1         # Parallelism level
```

### Run on Different Input

Update in config:
```yaml
input_file: /path/to/your/data.jsonl
```

### Adjust DeepSeek Server Args

In `assess-non-proof-deepseek.yaml`:
```yaml
stages:
  assess_complete_solution_quality:
    server_args: "--ep-size 16 --dp 16 ..."  # Adjust as needed
```

---

## 📝 Related Files

**Pipelines:**
- `pipeline/assess_non_proof_pipeline.py` (120b version)
- `pipeline/assess_non_proof_pipeline-deepseek.py` (DeepSeek version)

**Configs:**
- `configs/assess-non-proof-solutions.yaml` (120b config)
- `configs/assess-non-proof-deepseek.yaml` (DeepSeek config)

**Prompt:**
- `prompts/assess-complete-solution-quality.yaml`

**Post-processing:**
- `scripts/postprocess_quality_assessment.py`

---

## 💡 Tips

1. **Start small**: Test with `num_chunks: 1` on a small sample first
2. **Monitor logs**: Check `generation-logs/` for any inference issues
3. **Acceptance rate**: Typically 60-80% for clean data
4. **Key naming**: Ensure your data has `extracted_solution` (not `solution` or `generation`)

---

## 🐛 Troubleshooting

**Error: "Key 'extracted_solution' not found"**
- Your data uses `solution` or `generation` instead
- Update your data or modify the prompt to use the correct key

**Error: "FileNotFoundError"**
- Check `input_file` path in config
- Ensure the file exists and is accessible

**Low acceptance rate (<40%)**
- Review rejected samples in `output.jsonl`
- Check `reasoning_content` for common rejection reasons
- May need to improve upstream data quality

---

## 🚀 Next Steps

After assessment:
1. Use `accepted.jsonl` for RL training
2. Analyze `rejected.jsonl` to improve data pipeline
3. Iterate on filtering/extraction stages if needed
