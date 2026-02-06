"""
SuperV3 GRPO training test script.
Aligned with super_nemo_rl/run_debug_superv3.sh

Usage:
    python scripts/test_super_branch.py
"""
from nemo_skills.pipeline.cli import grpo_gym_nemo_rl, wrap_arguments


# Cluster and model config
cluster = 'dfw'
backend = 'megatron'

num_gpu = {
    'dfw': 8,
    'eos': 8,
    'hsg': 4,
    'lax': 8,
}

models = {
    'dfw': {
        'super': '/lustre/fs1/portfolios/coreai/users/yifuw/checkpoints/nemotron-super-sft-repeated-mtp-iter-0010600',
    },
    'lax': {
        'super': '/lustre/fs1/portfolios/coreai/users/yifuw/checkpoints/nemotron-super-sft-repeated-mtp-iter-0010600',
    },
}

train_data = {
    'dfw': '/lustre/fsw/portfolios/llmservice/users/wedu/data/rl_data/dapo_cot.jsonl',
    'lax': '/lustre/fsw/portfolios/llmservice/users/wedu/data/rl_data/dapo_cot.jsonl',
    'eos': '/lustre/fsw/llmservice_nemo_reasoning/wedu/data/grpo-gym/train.jsonl',
}

val_data = {
    'dfw': '/lustre/fsw/portfolios/llmservice/users/wedu/data/rl_data/dapo_cot.jsonl',
    'lax': '/lustre/fsw/portfolios/llmservice/users/wedu/data/rl_data/dapo_cot.jsonl',
    'eos': '/lustre/fsw/llmservice_nemo_reasoning/wedu/data/grpo-gym/val.jsonl',
}


# Training params (debug config, aligned with run_debug_superv3.sh)
num_prompts_per_step = 8       # PPS
num_generations_per_prompt = 4 # GPP
train_global_batch_size = 32   # GBS = PPS * GPP (required by force_on_policy_ratio=True)
max_sequence_length = 2048
max_num_steps = 20
lr_warmup_iters = 0

# Parallel config for 4 nodes (32 GPUs)
# Reference: yifuw's config TP=8, CP=8, EP=64, PP=1 for 20 nodes (160 GPUs)
TP = 8
EP = 32
ETP = 1
CP = 4
PP = 1

# Other config
sequence_packing = True
token_level_loss = True
kl_penalty = 0
clip_min = 0.2
clip_max = 0.28
temperature = 1.0
lr = 3e-6

val_period = 500
save_period = 100
keep_top_k = 1

# vLLM config
vllm_tp = TP
vllm_gpu_memory_util = 0.6
vllm_enforce_eager = False
colocated_enabled = True

# Resources
num_nodes = 4
train_micro_batch_size = 1
logprob_batch_size = 1

# Output
wandb_project = f'grpo-superv3-bihu'
expname = f'bihu-superv3-t{TP}e{EP}c{CP}p{PP}-pps{num_prompts_per_step}gpp{num_generations_per_prompt}gbs{train_global_batch_size}-lr{lr}-len{max_sequence_length}'
output_dir = f'/lustre/fsw/portfolios/llmservice/users/bihu/experiments/grpo/{expname}'

print(f"Experiment: {expname}")
print(f"Output: {output_dir}")
print(f"Cluster: {cluster} ({num_nodes} nodes x {num_gpu[cluster]} GPUs)")
print(f"Model: {models[cluster]['super']}")
print(f"Parallel: TP={TP}, EP={EP}, ETP={ETP}, CP={CP}, PP={PP}")
print(f"Batch: PPS={num_prompts_per_step}, GPP={num_generations_per_prompt}, GBS={train_global_batch_size}")


grpo_gym_nemo_rl(
    ctx=wrap_arguments(
        # Config file
        '--config=/nemo_run/code/nemo_skills/training/nemo_rl/configs/grpo_superv3.yaml '
        # GRPO params
        f'grpo.max_num_steps={max_num_steps} '
        f'grpo.num_prompts_per_step={num_prompts_per_step} '
        f'grpo.num_generations_per_prompt={num_generations_per_prompt} '
        f'grpo.val_period={val_period} '
        f'grpo.val_at_start=False '
        f'grpo.async_grpo.enabled=False '
        f'grpo.seq_logprob_error_threshold=null '
        # Policy params
        f'policy.max_total_sequence_length={max_sequence_length} '
        f'policy.train_global_batch_size={train_global_batch_size} '
        f'policy.train_micro_batch_size={train_micro_batch_size} '
        f'policy.logprob_batch_size={logprob_batch_size} '
        f'policy.sequence_packing.enabled={str(sequence_packing)} '
        # Megatron parallel config
        f'policy.megatron_cfg.tensor_model_parallel_size={TP} '
        f'policy.megatron_cfg.expert_tensor_parallel_size={ETP} '
        f'policy.megatron_cfg.expert_model_parallel_size={EP} '
        f'policy.megatron_cfg.context_parallel_size={CP} '
        f'policy.megatron_cfg.pipeline_model_parallel_size={PP} '
        f'policy.megatron_cfg.sequence_parallel=True '
        f'policy.megatron_cfg.activation_checkpointing=True '
        f'policy.megatron_cfg.bias_activation_fusion=False '
        f'policy.megatron_cfg.moe_permute_fusion=True '
        f'policy.megatron_cfg.defer_fp32_logits=True '
        f'policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=False '
        # MTP config
        f'policy.megatron_cfg.mtp_loss_scaling_factor=0.0 '
        f'policy.megatron_cfg.mtp_use_repeated_layer=true '
        # Error path
        f'policy.megatron_cfg.error_path={output_dir}/checkpoints/error_logs '
        # Optimizer
        f'policy.megatron_cfg.optimizer.lr={lr} '
        f'policy.megatron_cfg.optimizer.weight_decay=0 '
        f'policy.megatron_cfg.scheduler.lr_warmup_iters={lr_warmup_iters} '
        # vLLM generation config
        f'policy.generation.temperature={temperature} '
        f'policy.generation.colocated.enabled={str(colocated_enabled)} '
        f'policy.generation.colocated.resources.num_nodes=0 '
        f'policy.generation.colocated.resources.gpus_per_node=8 '
        f'policy.generation.vllm_cfg.tensor_parallel_size={vllm_tp} '
        f'policy.generation.vllm_cfg.gpu_memory_utilization={vllm_gpu_memory_util} '
        f'policy.generation.vllm_cfg.enforce_eager={str(vllm_enforce_eager)} '
        # New params (use ++ to add)
        f'++policy.generation.vllm_cfg.skip_tokenizer_init=False '
        # Delete speculative_config (use ~ to remove)
        f'~policy.generation.vllm_kwargs.speculative_config '
        # Loss config
        f'loss_fn.reference_policy_kl_penalty={kl_penalty} '
        f'loss_fn.ratio_clip_min={clip_min} '
        f'loss_fn.ratio_clip_max={clip_max} '
        f'loss_fn.force_on_policy_ratio=True '
        f'loss_fn.use_importance_sampling_correction=True '
        f'loss_fn.token_level_loss={str(token_level_loss).lower()} '
        # Checkpointing
        f'checkpointing.save_period={save_period} '
        f'checkpointing.keep_top_k={keep_top_k} '
    ),
    cluster=cluster,
    wandb_project=wandb_project,
    expname=expname,
    backend=backend,
    output_dir=output_dir,
    hf_model=models[cluster]['super'],
    num_nodes=num_nodes,
    num_gpus=num_gpu[cluster],
    training_data=train_data[cluster],
    validation_data=val_data[cluster],
    disable_wandb=True,
    with_sandbox=False,
)

print(f"Job submitted: {expname}")
