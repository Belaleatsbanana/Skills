#!/usr/bin/env python3
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
Non-Proof Problem Assessment Pipeline - DeepSeek Version

This pipeline assesses non-proof problems (natural language math problems)
with solutions for RL training quality using DeepSeek models.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, wrap_arguments


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def assess_complete_solution_quality(cluster, expname, run_after, stage_config, **kwargs):
    """Assess complete problem-answer-solution triplet quality - returns ACCEPT/REJECT decision."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_quality_assessment.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/accepted.jsonl "
        f"    {output_dir}/rejected.jsonl "
        f"    --stage complete_solution_quality "
    )

    # Get server args from stage config with deepseek defaults
    server_args = stage_config.get(
        "server_args",
        "--ep-size 16 --dp 16 --enable-dp-attention --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 --mem-fraction-static=0.8",
    )

    stage_kwargs = stage_config.get("stage_kwargs", {}).copy()
    stage_kwargs["server_args"] = server_args

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/non-proof/assess-problem-answer-quality.yaml "
            f"++inference.top_p=0.95 "
            f"++inference.temperature=1.0 "
            f"++inference.tokens_to_generate=120000 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        **stage_kwargs,
    )


stages_map = {
    "assess_complete_solution_quality": assess_complete_solution_quality,
}


def get_available_configs(config_dir):
    """Get available YAML configuration files from the config directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    yaml_files = list(config_dir.glob("*.yaml"))
    config_names = [file.stem for file in yaml_files if not file.name.startswith("template")]
    return config_names


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "configs"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="Non-Proof Problem Assessment Pipeline - DeepSeek")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=available_configs,
        help="Config file to use (from rl-data-clean/configs/)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )

    args = parser.parse_args()

    config_path = config_dir / f"{args.config}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    if args.stages:
        # Stages specified via command line
        stages_to_run = args.stages.split(",")
        print(f"Running specified stages: {stages_to_run}")
    else:
        # No command line override, run all stages from config
        stages_to_run = full_stage_sequence
        print(f"Running all stages defined in config '{args.config}': {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage specified: '{stage}'. Available stages: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(
                f"Stage '{stage}' requested but not part of the defined sequence in {config_path}. "
                f"Specify one of {full_stage_sequence}."
            )

    # --- Common parameters ---
    base_output_dir = config["base_output_dir"]
    suffix = config.get("suffix", args.config)
    cluster = config["cluster"]
    expname_base = config["expname"]

    # --- Run selected stages ---
    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get("stages", {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)

        # Determine dependencies
        dependencies_list = stage_config.get("dependencies", [])
        if dependencies_list:
            run_after = [get_stage_expname(expname_base, dep, suffix) for dep in dependencies_list]
        else:
            run_after = None

        stage_func(
            cluster=cluster,
            expname=current_expname,
            run_after=run_after,
            stage_config=stage_config,
        )

    print("\n=== Pipeline execution completed! ===")
