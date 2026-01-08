#!/bin/bash
# Test script for GRPO-Gym with NeMo Skills ns_tools (Python execution)
# This tests stateful Python code execution with math verification

set -e

echo "=================================="
echo "GRPO-Gym + ns_tools Test"
echo "=================================="

# Configuration
MODEL_PATH="${NEMO_SKILLS_TEST_HF_MODEL:-/home/wedu/Qwen3-0.6B}"
MODEL_TYPE="${NEMO_SKILLS_TEST_MODEL_TYPE:-qwen}"
OUTPUT_DIR="/tmp/grpo-gym-ns-tools-test-$(date +%s)"
BACKEND="fsdp"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Backend: $BACKEND"
echo "  Output: $OUTPUT_DIR"
echo ""

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GYM_DIR="/home/wedu/Gym"

echo ""
echo "📂 Directories:"
echo "  NeMo-Skills: $PROJECT_DIR"
echo "  Gym: $GYM_DIR"
echo ""

# Check if Gym ns_tools branch is available
if [ ! -d "$GYM_DIR/resources_servers/ns_tools" ]; then
    echo "❌ Error: ns_tools not found in $GYM_DIR"
    echo "Please checkout George's branch first:"
    echo "  cd $GYM_DIR"
    echo "  git remote add gwarmstrong https://github.com/gwarmstrong/Gym.git"
    echo "  git fetch gwarmstrong georgea/add-nemo-skills-tool-resource"
    echo "  git checkout -b test-ns-tools gwarmstrong/georgea/add-nemo-skills-tool-resource"
    exit 1
fi

# Use George's example data (already in correct format)
echo "📝 Using George's example data..."
mkdir -p "$PROJECT_DIR/tests/data"

# Copy first 2 examples from George's data for quick test (from host Gym repo)
head -2 "$GYM_DIR/resources_servers/ns_tools/data/example.jsonl" > "$PROJECT_DIR/tests/data/ns-tools-sample.jsonl"

echo "✅ Created: $PROJECT_DIR/tests/data/ns-tools-sample.jsonl"
echo ""

# Run GRPO-Gym training with ns_tools
echo "🚀 Starting GRPO-Gym training with ns_tools..."
echo ""

python -c "
from pathlib import Path
from nemo_skills.pipeline.cli import grpo_gym_nemo_rl, wrap_arguments

grpo_gym_nemo_rl(
    ctx=wrap_arguments(
        '++grpo.max_num_steps=2 '
        '++grpo.num_prompts_per_step=2 '
        '++policy.max_total_sequence_length=512 '
        '++policy.dtensor_cfg.tensor_parallel_size=1 '
        '++checkpointing.save_period=2 '
        '++policy.train_global_batch_size=2 '
        '++policy.train_micro_batch_size=1 '
        '++policy.optimizer.kwargs.lr=1e-6 '
        # Rewrite config_paths to include ns_tools (Hydra doesn't support += for lists)
        '++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/math_with_judge/configs/math_with_judge.yaml,/opt/nemo-rl/3rdparty/Gym-workspace/Gym/resources_servers/ns_tools/configs/ns_tools.yaml] '
        # Disable LLM judge (use rule-based math verification)
        '++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.should_use_judge=false '
    ),
    cluster='test-local',
    config_dir=Path('${SCRIPT_DIR}').absolute(),
    expname='test-grpo-gym-ns-tools',
    output_dir='${OUTPUT_DIR}',
    hf_model='${MODEL_PATH}',
    num_nodes=1,
    num_gpus=1,
    training_data='${PROJECT_DIR}/tests/data/ns-tools-sample.jsonl',
    validation_data='${PROJECT_DIR}/tests/data/ns-tools-sample.jsonl',
    backend='${BACKEND}',
    disable_wandb=True,
    with_sandbox=True,  # Enable sandbox for Python execution
)
"

echo ""
echo "=================================="
echo "✅ Test completed!"
echo "=================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To check the output:"
echo "  ls -lh $OUTPUT_DIR/training-logs/"
echo "  cat $OUTPUT_DIR/training-logs/*.out"
