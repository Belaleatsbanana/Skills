# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pipeline command for GRPO training with NemoGym (no proxy server).
For quick debugging and simple task evaluation.
"""

import inspect
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.nemo_rl import nemo_rl_app
from nemo_skills.pipeline.utils import (
    add_task,
    check_if_mounted,
    check_mounts,
    get_cluster_config,
    get_env_variables,
    get_exp,
    get_free_port,
    get_mounted_path,
    get_nsight_cmd,
    get_timeout_str,
    parse_kwargs,
    resolve_mount_paths,
    run_exp,
    temporary_env_update,
)
from nemo_skills.utils import (
    get_logger_name,
    setup_logging,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))


class SupportedBackends(str, Enum):
    fsdp = "fsdp"
    megatron = "megatron"


@dataclass
class NemoRLGymTask:
    model: str
    output_dir: str
    train_data: str
    val_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    disable_wandb: bool
    wandb_project: str
    wandb_group: str
    timeout: str
    log_dir: str
    env_variables: dict
    backend: str
    profile_step_range: str
    extra_arguments: str = ""

    def format_train_args(self):
        cmd = (
            f"++policy.model_name={self.model} "
            f"++cluster.gpus_per_node={self.num_gpus} "
            f"++cluster.num_nodes={self.num_nodes} "
            f"++checkpointing.checkpoint_must_save_by={self.timeout} "
            f"++logger.log_dir={self.log_dir} "
            f"++checkpointing.checkpoint_dir={self.output_dir}/checkpoints "
        )
        if self.backend == "megatron":
            cmd += " ++policy.dtensor_cfg.enabled=false ++policy.megatron_cfg.enabled=true "
            cmd += " ++policy.optimizer=None ++policy.dynamic_batching.enabled=false "
        else:
            cmd += " ++policy.dtensor_cfg.enabled=true ++policy.megatron_cfg.enabled=false "

        return cmd

    def format_data_args(self):
        cmd = f"++data.train_jsonl_fpath={self.train_data} "
        if self.val_data is not None:
            cmd += f"++data.validation_jsonl_fpath={self.val_data} "
        return cmd

    def format_wandb_args(self):
        wandb_id = self.expname + ("-" + self.wandb_group if self.wandb_group else "") + "-" + self.wandb_project
        cmd = (
            f"++logger.wandb_enabled={not self.disable_wandb} "
            f"++logger.wandb.project={self.wandb_project} "
            f"++logger.wandb.name={self.expname} "
            f"++logger.wandb.id={wandb_id} "
        )
        if self.wandb_group:
            cmd += f"++logger.wandb.group={self.wandb_group} "

        if not self.disable_wandb:
            validate_wandb_project_name(
                wandb_project=self.wandb_project,
                wandb_name=self.expname,
                wandb_group=self.wandb_group,
                wandb_id=wandb_id,
            )

        return cmd

    def get_cmd(self):
        self.logging_params = self.format_wandb_args()
        nsight_cmd = get_nsight_cmd(self.profile_step_range)
        cmd = (
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code:/opt/nemo-rl && "
            f"export UV_PROJECT=/opt/nemo-rl && "
            f"{nsight_cmd}"
            f"echo 'Starting GRPO training with NemoGym (proof)' && "
            f"uv run python /opt/nemo-rl/examples/nemo_gym/run_grpo_proof.py "
            f"  --config /nemo_run/code/nemo_skills/training/nemo_rl/configs/grpo_gym.yaml "
            f"  {self.format_train_args()} "
            f"  {self.format_data_args()} "
            f"  {self.logging_params} "
            f"  {self.extra_arguments} "
        )
        return cmd


def get_training_cmd(
    cluster_config,
    partition,
    hf_model,
    output_dir,
    train_data,
    val_data,
    num_gpus,
    num_nodes,
    expname,
    disable_wandb,
    wandb_project,
    wandb_group,
    extra_arguments,
    log_dir,
    env_variables,
    backend,
    profile_step_range,
):
    timeout = get_timeout_str(cluster_config, partition)

    task = NemoRLGymTask(
        model=hf_model,
        output_dir=output_dir,
        train_data=train_data,
        val_data=val_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        timeout=timeout,
        extra_arguments=extra_arguments,
        log_dir=log_dir,
        env_variables=env_variables,
        backend=backend,
        profile_step_range=profile_step_range,
    )

    return task.get_cmd()


def get_checkpoint_convert_cmd(output_dir, final_hf_path, step, backend, max_position_embeddings=None):
    cmd = "export PYTHONPATH=$PYTHONPATH:/nemo_run/code && export UV_PROJECT=/opt/nemo-rl && cd /nemo_run/code && "
    if backend == "fsdp":
        cmd += "uv run --extra automodel python -m nemo_skills.training.nemo_rl.convert_dcp_to_hf "
    elif backend == "megatron":
        cmd += "uv run --extra mcore python -m nemo_skills.training.nemo_rl.convert_megatron_to_hf "
    else:
        raise ValueError("Invalid backend: must be 'fsdp' or 'megatron'")

    cmd += f" --training-folder={output_dir} "
    cmd += f" --hf-ckpt-path={final_hf_path} "
    if max_position_embeddings is not None:
        cmd += f" --max-position-embeddings={max_position_embeddings} "
    if step != "last":
        try:
            step = int(step)
        except ValueError:
            raise ValueError(
                f"Invalid step value: {step}. Expected a string representing an integer (e.g. '100') or 'last'."
            )
        cmd += f" --step {step} "

    return cmd


@nemo_rl_app.command(name="grpo-gym", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def grpo_gym_nemo_rl(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    final_hf_path: str = typer.Option(
        None,
        help="If specified, will save the final HF model to this path. "
        "If not specified, will save to output_dir/final_hf_model",
    ),
    expname: str = typer.Option("grpo-gym", help="Nemo run experiment name"),
    hf_model: str = typer.Option(..., help="Path to the HF model"),
    training_data: str = typer.Option(..., help="Path to the NemoGym training data (JSONL)"),
    validation_data: str = typer.Option(..., help="Path to the NemoGym validation data (JSONL)"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs per node"),
    conversion_step: str = typer.Option(
        default="last",
        help=(
            "The checkpoint step to convert. Use 'last' (default) to convert the latest checkpoint, "
            "or specify a step number explicitly, e.g. --conversion-step '100'."
        ),
    ),
    run_conversion_only: bool = typer.Option(False, help="Whether to run checkpoint conversion only or not."),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    wandb_group: str = typer.Option(None, help="Weights & Biases group name."),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    profile_step_range: str = typer.Option(
        None,
        help="Controls which training steps the nsys profiler captures. "
        "Format: START:STOP (1-indexed, STOP exclusive, same as slice syntax arr[start:stop]). "
        "Example: '3:5' profiles steps 3 and 4 only. NOTE: START must be ≥ 1, so '0:10' is invalid.",
    ),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    backend: SupportedBackends = typer.Option(
        SupportedBackends.fsdp,
        "--backend",
        help="Choose backend. Supported options: fsdp, megatron",
    ),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    skip_hf_home_check: bool | None = typer.Option(
        None,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    sbatch_kwargs: str = typer.Option(
        "",
        help="Additional sbatch kwargs to pass to the job scheduler. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
    max_position_embeddings: int = typer.Option(
        None,
        help="Max position embeddings to use for conversion. If not specified, will be inferred from the model config.",
    ),
    # LLM-as-a-judge: start a judge server in the same heterogeneous job
    server_model: Optional[str] = typer.Option(
        None,
        help="Model path/name for the judge server (e.g. HF model name). If set, starts a judge server in a heterogeneous job so math_with_judge can use it.",
    ),
    server_type: str = typer.Option(
        "vllm",
        help="How to serve the judge model: 'vllm', 'sglang', 'trtllm', or 'generic'. Must have a matching key under cluster config 'containers'.",
    ),
    server_gpus: int = typer.Option(1, help="Number of GPUs per node for the judge server."),
    server_nodes: int = typer.Option(1, help="Number of nodes for the judge server."),
    server_args: str = typer.Option(
        "", help="Extra arguments passed to the judge server command (e.g. vLLM/sglang CLI flags)."
    ),
):
    """Runs NeMo-RL GRPO training with NemoGym integration (no proxy server).

    Simplified setup for quick debugging and simple task evaluation.
    All extra arguments are passed directly to the NeMo-RL GRPO-Gym script.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting GRPO-Gym training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    if log_dir is None:
        log_dir = output_dir

    output_dir, log_dir = check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if hf_model.startswith("/"):
        check_if_mounted(cluster_config, hf_model)

    training_data = get_mounted_path(cluster_config, training_data)
    validation_data = get_mounted_path(cluster_config, validation_data)

    env_variables = get_env_variables(cluster_config)
    if backend == "megatron":
        if "HF_HOME" not in env_variables:
            raise typer.BadParameter(
                "Missing required environment variable 'HF_HOME' for 'megatron' backend.\n"
                "You can set it in your cluster config like this:\n"
                '  env_vars: ["HF_HOME=/your/path/to/hf_home"]'
            )

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=partition,
        hf_model=hf_model,
        output_dir=output_dir,
        train_data=training_data,
        val_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        extra_arguments=extra_arguments,
        log_dir=f"{log_dir}/training-logs",
        env_variables=env_variables,
        backend=backend,
        profile_step_range=profile_step_range,
    )

    # Build judge server config for heterogeneous job (LLM-as-a-judge for math_with_judge)
    server_config = None
    if server_model:
        if not server_gpus or server_gpus < 1:
            raise typer.BadParameter("server_model requires server_gpus >= 1.")
        server_port = get_free_port(strategy="random")
        server_config = {
            "server_type": server_type,
            "num_gpus": server_gpus,
            "num_nodes": server_nodes,
            "model_path": server_model,
            "server_port": server_port,
            "server_args": server_args or "",
            "n_servers": 1,
        }
        LOG.info("Judge server will be started in heterogeneous job: %s", json.dumps(server_config, indent=2))

    env_update = {"RAY_LOG_SYNC_FREQUENCY": 20} if profile_step_range else {}
    sbatch_kwargs = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    # When with_ray + judge: start nemo-rl (main) first so Ray head runs in nemo-rl container (has ray[client]).
    add_task_params = inspect.signature(add_task).parameters
    server_goes_first_override = False if (server_config is not None) else None
    extra_add_task_kwargs = {}
    if "server_goes_first_override" in add_task_params:
        extra_add_task_kwargs["server_goes_first_override"] = server_goes_first_override
    if "n_servers" in add_task_params and server_config is not None:
        extra_add_task_kwargs["n_servers"] = server_config.get("n_servers", 1)

    # Inject JUDGE_SERVER_ARGS into the main task command so the process has it even if executor
    # env_vars are not passed to the container (e.g. some het job setups). math_with_judge needs it.
    if server_config is not None:
        judge_het_group = 1  # main task first (server_goes_first_override=False) → judge is het group 1
        judge_server_args = json.dumps(
            {
                "server_type": server_config["server_type"],
                "model": server_config["model_path"],
                "port": server_config["server_port"],
                "n_servers": server_config.get("n_servers", 1),
                "judge_het_group": judge_het_group,
            }
        )
        # Escape for shell single-quoted string: ' -> '\''
        escaped = judge_server_args.replace("'", "'\"'\"'")
        math_judge_log_dir = f"{log_dir}/training-logs"
        escaped_log_dir = math_judge_log_dir.replace("'", "'\"'\"'")
        train_cmd = (
            f"export JUDGE_SERVER_ARGS='{escaped}' && export MATH_JUDGE_LOG_DIR='{escaped_log_dir}' && {train_cmd}"
        )
        LOG.info(
            "Prefixed train_cmd with export JUDGE_SERVER_ARGS and MATH_JUDGE_LOG_DIR=%s for main task.",
            math_judge_log_dir,
        )

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        prev_task = _task_dependencies
        with temporary_env_update(cluster_config, env_update):
            if not run_conversion_only:
                prev_task = add_task(
                    exp,
                    cmd=train_cmd,
                    task_name=f"{expname}-grpo-gym",
                    log_dir=f"{log_dir}/training-logs",
                    container=cluster_config["containers"]["nemo-rl"],
                    num_gpus=num_gpus,
                    num_nodes=num_nodes,
                    cluster_config=cluster_config,
                    server_config=server_config,
                    partition=partition,
                    run_after=run_after,
                    reuse_code=reuse_code,
                    reuse_code_exp=reuse_code_exp,
                    task_dependencies=[prev_task] if prev_task is not None else None,
                    sbatch_kwargs=sbatch_kwargs,
                    heterogeneous=True if server_config is not None else False,
                    with_sandbox=with_sandbox,
                    with_ray=True,
                    installation_command=installation_command,
                    skip_hf_home_check=skip_hf_home_check,
                    **extra_add_task_kwargs,
                )

        prev_task = add_task(
            exp,
            cmd=get_checkpoint_convert_cmd(
                output_dir=output_dir,
                final_hf_path=final_hf_path or f"{output_dir}/final_hf_model",
                step=conversion_step,
                backend=backend,
                max_position_embeddings=max_position_embeddings,
            ),
            task_name=f"{expname}-convert-final-ckpt",
            log_dir=f"{log_dir}/convert-final-ckpt",
            container=cluster_config["containers"]["nemo-rl"],
            cluster_config=cluster_config,
            partition=partition,
            num_nodes=1,
            num_tasks=1,
            num_gpus=num_gpus,
            run_after=run_after,
            reuse_code=reuse_code,
            reuse_code_exp=reuse_code_exp,
            task_dependencies=[prev_task] if prev_task is not None else None,
            sbatch_kwargs=sbatch_kwargs,
            installation_command=installation_command,
            skip_hf_home_check=skip_hf_home_check,
        )

        run_exp(exp, cluster_config, sequential=False, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
