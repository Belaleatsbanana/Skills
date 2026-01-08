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

"""NeMo Gym Rollouts Pipeline.

This pipeline command runs rollout collection with NeMo Gym, orchestrating:
- vLLM model server (optional, can use pre-hosted)
- Sandbox for code execution (optional)
- NeMo Gym servers (ng_run)
- Rollout collection client (ng_collect_rollouts)

Example usage:
    # Self-hosted vLLM server
    ns nemo_gym_rollouts \\
        --cluster local \\
        --config_paths "ns_tools/configs/ns_tools.yaml,math_with_judge/configs/math_with_judge.yaml" \\
        --input_file data/example.jsonl \\
        --output_file data/rollouts.jsonl \\
        --model /path/to/model \\
        --server_type vllm \\
        --server_gpus 1 \\
        --with_sandbox \\
        +agent_name=ns_tools_simple_agent \\
        +limit=10 \\
        +num_samples_in_parallel=3

    # Pre-hosted server
    ns nemo_gym_rollouts \\
        --cluster local \\
        --config_paths "ns_tools/configs/ns_tools.yaml" \\
        --input_file data/example.jsonl \\
        --output_file data/rollouts.jsonl \\
        --server_address http://localhost:8000/v1 \\
        --server_type vllm \\
        +agent_name=ns_tools_simple_agent
"""

import logging

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.cluster import parse_kwargs
from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.scripts import (
    NemoGymRolloutsScript,
    SandboxScript,
    ServerScript,
)
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def nemo_gym_rollouts(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    config_paths: str = typer.Option(
        ...,
        help="Comma-separated list of NeMo Gym config YAML files for ng_run. "
        "E.g., 'ns_tools/configs/ns_tools.yaml,math_with_judge/configs/math_with_judge.yaml'",
    ),
    input_file: str = typer.Option(..., help="Path to input JSONL file for rollout collection"),
    output_file: str = typer.Option(..., help="Path to output JSONL file for rollouts"),
    expname: str = typer.Option("nemo_gym_rollouts", help="NeMo Run experiment name"),
    model: str = typer.Option(None, help="Path to model for self-hosted vLLM server"),
    server_address: str = typer.Option(
        None,
        help="Address of pre-hosted server (e.g., http://localhost:8000/v1). If provided, skips self-hosted server.",
    ),
    server_type: pipeline_utils.SupportedServers = typer.Option(
        None,
        help="Type of server (vllm, trtllm, sglang, etc.)",
    ),
    server_gpus: int = typer.Option(None, help="Number of GPUs for self-hosted server"),
    server_nodes: int = typer.Option(1, help="Number of nodes for self-hosted server"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    with_sandbox: bool = typer.Option(False, help="If True, start a sandbox container for code execution"),
    gym_path: str = typer.Option(
        "/opt/NeMo-RL/3rdparty/Gym-workspace/Gym",
        help="Path to NeMo Gym installation. Defaults to container built-in. Use for mounted/custom Gym.",
    ),
    policy_api_key: str = typer.Option(
        "dummy",
        help="API key for policy server. Use 'dummy' for local vLLM servers.",
    ),
    policy_model_name: str = typer.Option(
        None,
        help="Model name for policy server. Required for pre-hosted servers. "
        "For self-hosted, defaults to the model path if not specified.",
    ),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    qos: str = typer.Option(None, help="Specify Slurm QoS"),
    time_min: str = typer.Option(None, help="Slurm time-min parameter"),
    config_dir: str = typer.Option(None, help="Custom directory for cluster configs"),
    log_dir: str = typer.Option(None, help="Custom location for logs"),
    exclusive: bool | None = typer.Option(None, help="Add exclusive flag to slurm job"),
    dry_run: bool = typer.Option(False, help="Validate without executing"),
    sbatch_kwargs: str = typer.Option("", help="Additional sbatch kwargs as JSON string"),
):
    """Run NeMo Gym rollout collection pipeline.

    This command orchestrates running rollout collection with NeMo Gym:
    1. Optionally starts a vLLM model server (or uses pre-hosted)
    2. Optionally starts a sandbox for code execution
    3. Starts NeMo Gym servers via ng_run
    4. Runs ng_collect_rollouts to collect rollouts

    All Hydra arguments (prefixed with + or ++) are passed through to ng_run
    and ng_collect_rollouts. Common arguments include:
    - +agent_name=... (required for rollout collection)
    - +limit=... (limit number of samples)
    - +num_samples_in_parallel=... (concurrent requests)
    - ++policy_model_name=... (model name override)
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = " ".join(ctx.args)
    LOG.info("Starting NeMo Gym rollouts pipeline")
    LOG.info(f"Extra arguments: {extra_arguments}")

    # Parse config paths
    config_paths_list = [p.strip() for p in config_paths.split(",") if p.strip()]
    LOG.info(f"Config paths: {config_paths_list}")

    # Validate server configuration
    self_hosted = model is not None and server_gpus is not None
    pre_hosted = server_address is not None

    if not self_hosted and not pre_hosted:
        raise ValueError(
            "Must provide either --model and --server_gpus for self-hosted server, "
            "or --server_address for pre-hosted server"
        )

    if self_hosted and pre_hosted:
        raise ValueError("Cannot specify both self-hosted (--model, --server_gpus) and pre-hosted (--server_address)")

    if self_hosted and server_type is None:
        raise ValueError("--server_type is required when using self-hosted server")

    # Validate and set policy_model_name
    if pre_hosted and policy_model_name is None:
        raise ValueError("--policy_model_name is required when using a pre-hosted server (--server_address)")

    if self_hosted and policy_model_name is None:
        # For self-hosted, default to the model path
        policy_model_name = model
        LOG.info(f"Using model path as policy_model_name: {policy_model_name}")

    # Get cluster config
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)

    if not log_dir:
        log_dir = f"{output_file.rsplit('/', 1)[0]}/logs" if "/" in output_file else "./logs"

    # Parse sbatch kwargs
    sbatch_kwargs_dict = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    # Build pipeline components
    components = []
    server_script = None
    sandbox_script = None

    # 1. Server (optional, self-hosted)
    if self_hosted:
        server_type_str = server_type.value if hasattr(server_type, "value") else server_type
        server_container = cluster_config["containers"].get(server_type_str, server_type_str)

        server_script = ServerScript(
            server_type=server_type_str,
            model_path=model,
            cluster_config=cluster_config,
            num_gpus=server_gpus,
            num_nodes=server_nodes,
            server_args=server_args,
            allocate_port=True,
        )

        server_cmd = Command(
            script=server_script,
            container=server_container,
            name=f"{expname}_server",
        )
        components.append(server_cmd)
        LOG.info(f"Added self-hosted {server_type_str} server on port {server_script.port}")

    # 2. Sandbox (optional)
    if with_sandbox:
        sandbox_script = SandboxScript(
            cluster_config=cluster_config,
            allocate_port=True,
        )

        sandbox_cmd = Command(
            script=sandbox_script,
            container=cluster_config["containers"]["sandbox"],
            name=f"{expname}_sandbox",
        )
        components.append(sandbox_cmd)
        LOG.info(f"Added sandbox on port {sandbox_script.port}")

    # 3. NeMo Gym rollouts (ng_run + ng_status wait + ng_collect_rollouts)
    nemo_gym_script = NemoGymRolloutsScript(
        config_paths=config_paths_list,
        input_file=input_file,
        output_file=output_file,
        extra_arguments=extra_arguments,
        server=server_script,
        server_address=server_address,
        sandbox=sandbox_script,
        gym_path=gym_path,
        policy_api_key=policy_api_key,
        policy_model_name=policy_model_name,
    )

    nemo_gym_cmd = Command(
        script=nemo_gym_script,
        container=cluster_config["containers"]["nemo-rl"],
        name=f"{expname}_nemo_gym",
    )
    components.append(nemo_gym_cmd)
    LOG.info("Added NeMo Gym rollouts command (ng_run + ng_collect_rollouts)")

    # Create command group
    hardware = HardwareConfig(
        partition=partition,
        num_gpus=server_gpus if self_hosted else 0,
        num_nodes=server_nodes if self_hosted else 1,
        num_tasks=1,
        sbatch_kwargs=sbatch_kwargs_dict,
    )

    group = CommandGroup(
        commands=components,
        hardware=hardware,
        name=expname,
        log_dir=log_dir,
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=[{"name": expname, "group": group}],
    )

    sequential = cluster_config["executor"] in ["local", "none"]
    result = pipeline.run(dry_run=dry_run, sequential=sequential)

    LOG.info(f"Pipeline {'validated' if dry_run else 'submitted'} successfully")
    return result


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
