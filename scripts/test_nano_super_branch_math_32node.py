from nemo_skills.pipeline.cli import grpo_gym_nemo_rl, wrap_arguments


# Cluster and model
cluster = "dfw"
model_size = "31B"
backend = "megatron"
suffix = ""

# Cluster-specific configurations
num_gpu = {
    "dfw": 8,
    "eos": 8,
    "hsg": 4,
    "lax": 8,
}

models = {
    "dfw": {
        "7B": "/lustre/fsw/portfolios/llmservice/users/wedu/models/DeepSeek-R1-Distill-Qwen-7B",
        "30B": "/lustre/fsw/portfolios/llmservice/users/wedu/models/Qwen3-30B-A3B",
        "31B": "/lustre/fsw/portfolios/llmservice/users/soumyes/nano-sft-runs/eval_and_sleep/v3-sft-64gbs-nickel-capybara-5e-5-constant-wd-0-load-bal-1e-4-lcx3-pretool-base-temp1/iter_0013600/hf",
        "8B": "/hf_models/Qwen3-8B",
    },
    "eos": {
        "7B": "/lustre/fsw/llmservice_nemo_reasoning/wedu/models/Qwen/Qwen2.5-Math-7B-Instruct",
        "30B": "/lustre/fsw/llmservice_nemo_reasoning/wedu/models/Qwen3-30B-A3B",
    },
    "hsg": {
        "7B": "/lustre/fsw/portfolios/llmservice/users/wedu/models/Qwen/Qwen2.5-Math-7B-Instruct",
    },
    "lax": {
        "31B": "/lustre/fsw/portfolios/llmservice/users/wedu/models/iter_0013600",
    },
}

train_data = {
    "dfw": "/lustre/fsw/portfolios/llmservice/users/bihu/nanov3_math/data/curriculum_v7_acrid-teal.dapo17k.jsonl",
}

val_data = {
    "dfw": "/lustre/fsw/portfolios/llmservice/users/bihu/nanov3_math/data/curriculum_v7_acrid-teal.val.jsonl",
}

# ================ 训练参数配置 (对齐 run_grpo_math.sh) ================
num_prompts_per_step = 128  # PPS
num_generations_per_prompt = 16  # GPP
train_global_batch_size = 2048  # GBS
max_sequence_length = 49152  # SEQLEN
max_num_steps = 1_000_000

# 并行配置 (TP=4, EP=8, CP=4, PP=1)
TP = 4
EP = 8
CP = 4
PP = 1

# 训练参数
sequence_packing = True
token_level_loss = True
sequence_level_importance_ratios = False
kl_penalty = 0
clip_min = 0.2
clip_max = 0.28
temperature = 1.0
normalize_rewards = True
lr = 3e-6
use_on_policy_kl_approximation = True
use_importance_sampling_correction = True

use_best_at_k = False
best_at_k_k = 4
best_at_k_m = 10000

save_period = 5
val_period = 5
keep_top_k = 100

seq_logprob_error_threshold = 2
overlong_filtering = True

# MoE 相关
moe_freeze_router = True
moe_permute_fusion = True
moe_enable_deepep = False
moe_token_dispatcher_type = "alltoall"
moe_aux_loss_coeff = 0
moe_router_load_balancing_type = "none"
moe_router_bias_update_rate = 0

# vLLM 配置
vllm_tp = 4
vllm_gpu_memory_util = 0.5
vllm_enforce_eager = False
async_engine = True
colocated_enabled = True

# Resources
num_nodes = 32
train_micro_batch_size = 1

# Output (resume existing run)
wandb_project = "nano-v3-math"
expname = "rl-cot-31B-megatron-t4e8c4p1-pps128gpp16-seq49152"
output_dir = "/lustre/fsw/portfolios/llmservice/users/bihu/experiments/grpo-math/rl-cot-31B-megatron-t4e8c4p1-pps128gpp16-seq49152"

print(f":bar_chart: Experiment: {expname}")
print(f":file_folder: Output: {output_dir}")
print(f":desktop_computer:  Cluster: {cluster} ({num_nodes} nodes × {num_gpu[cluster]} GPUs)")
print(f":robot_face: Model: {models[cluster][model_size]}")

grpo_gym_nemo_rl(
    ctx=wrap_arguments(
        # ================ GRPO 参数 ================
        f"++grpo.max_num_steps={max_num_steps} "
        f"++grpo.num_prompts_per_step={num_prompts_per_step} "
        f"++grpo.num_generations_per_prompt={num_generations_per_prompt} "
        f"++grpo.val_period={val_period} "
        f"++grpo.val_at_start=false "
        f"++grpo.normalize_rewards={str(normalize_rewards).lower()} "
        f"++grpo.overlong_filtering={str(overlong_filtering).lower()} "
        f"++grpo.use_best_at_k={str(use_best_at_k).lower()} "
        f"++grpo.best_at_k_k={best_at_k_k} "
        f"++grpo.best_at_k_m={best_at_k_m} "
        f"++grpo.seq_logprob_error_threshold={seq_logprob_error_threshold} "
        f"++grpo.async_grpo.enabled=false "
        # ================ Policy 参数 ================
        f"++policy.max_total_sequence_length={max_sequence_length} "
        f"++policy.train_global_batch_size={train_global_batch_size} "
        f"++policy.train_micro_batch_size={train_micro_batch_size} "
        f"++policy.make_sequence_length_divisible_by=8 "
        f"++policy.dynamic_batching.enabled=false "
        f"++policy.sequence_packing.enabled={str(sequence_packing).lower()} "
        # ================ Megatron 并行配置 ================
        f"++policy.megatron_cfg.tensor_model_parallel_size={TP} "
        f"++policy.megatron_cfg.expert_tensor_parallel_size=1 "
        f"++policy.megatron_cfg.expert_model_parallel_size={EP} "
        f"++policy.megatron_cfg.context_parallel_size={CP} "
        f"++policy.megatron_cfg.pipeline_model_parallel_size={PP} "
        f"++policy.megatron_cfg.sequence_parallel=true "
        f"++policy.megatron_cfg.bias_activation_fusion=false "
        # ================ Megatron DDP 配置 ================
        f"++policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=false "
        # ================ MTP / MoE 配置 ================
        f"++policy.megatron_cfg.mtp_use_repeated_layer=true "
        f"++policy.megatron_cfg.empty_unused_memory_level=1 "
        f"++policy.megatron_cfg.moe_permute_fusion={str(moe_permute_fusion).lower()} "
        f"++policy.megatron_cfg.moe_enable_deepep={str(moe_enable_deepep).lower()} "
        f"++policy.megatron_cfg.moe_token_dispatcher_type={moe_token_dispatcher_type} "
        f"++policy.megatron_cfg.moe_aux_loss_coeff={moe_aux_loss_coeff} "
        f"++policy.megatron_cfg.moe_router_load_balancing_type={moe_router_load_balancing_type} "
        f"++policy.megatron_cfg.moe_router_bias_update_rate={moe_router_bias_update_rate} "
        f"++policy.megatron_cfg.freeze_moe_router={str(moe_freeze_router).lower()} "
        # ================ Megatron Scheduler/Optimizer ================
        f"++policy.megatron_cfg.scheduler.lr_warmup_iters=10 "
        f"++policy.megatron_cfg.scheduler.lr_warmup_init=0.0 "
        f"++policy.megatron_cfg.optimizer.lr={lr} "
        f"++policy.megatron_cfg.optimizer.weight_decay=0 "
        # ================ vLLM Generation 配置 ================
        f"++policy.generation.temperature={temperature} "
        f"++policy.generation.vllm_cfg.tensor_parallel_size={vllm_tp} "
        f"++policy.generation.vllm_cfg.gpu_memory_utilization={vllm_gpu_memory_util} "
        f"++policy.generation.vllm_cfg.enforce_eager={str(vllm_enforce_eager).lower()} "
        f"++policy.generation.vllm_cfg.async_engine={str(async_engine).lower()} "
        f"++policy.generation.colocated.enabled={str(colocated_enabled).lower()} "
        # ================ Loss 函数配置 ================
        f"++loss_fn.reference_policy_kl_penalty={kl_penalty} "
        f"++loss_fn.ratio_clip_min={clip_min} "
        f"++loss_fn.ratio_clip_max={clip_max} "
        f"++loss_fn.use_on_policy_kl_approximation={str(use_on_policy_kl_approximation).lower()} "
        f"++loss_fn.use_importance_sampling_correction={str(use_importance_sampling_correction).lower()} "
        f"++loss_fn.sequence_level_importance_ratios={str(sequence_level_importance_ratios).lower()} "
        f"++loss_fn.token_level_loss={str(token_level_loss).lower()} "
        # ================ Checkpointing 配置 ================
        f"++checkpointing.checkpoint_dir={output_dir} "
        f"++checkpointing.save_period={save_period} "
        f"++checkpointing.keep_top_k={keep_top_k} "
        f"++checkpointing.metric_name=train:total_reward/mean "
        f"++checkpointing.checkpoint_must_save_by=00:03:35:00 "
    ),
    cluster=cluster,
    wandb_project=wandb_project,
    expname=expname,
    backend=backend,
    output_dir=output_dir,
    hf_model=models[cluster][model_size],
    num_nodes=num_nodes,
    num_gpus=num_gpu[cluster],
    training_data=train_data[cluster],
    validation_data=val_data[cluster],
    disable_wandb=False,
    with_sandbox=False,
)

print(f":white_check_mark: Job submitted: {expname}")
