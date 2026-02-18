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
# WITHOUT WARRANTIES OR CONDITIONS OF KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Solve-then-assess problem quality pipeline (DeepSeek, proof).

Single-stage (combined): assess_problem_quality_with_solution only — model solves + judges in one call.
Optional two-stage: solve_proof_for_quality → assess_problem_quality_with_solution.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, wrap_arguments


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def _deepseek_stage_kwargs(stage_config):
    server_args = stage_config.get(
        "server_args",
        "--ep-size 16 --dp 16 --enable-dp-attention --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 --mem-fraction-static=0.8",
    )
    kw = stage_config.get("stage_kwargs", {}).copy()
    kw["server_args"] = server_args
    return kw


def solve_proof_for_quality(cluster, expname, run_after, stage_config, **kwargs):
    """Stage 1: Attempt to solve each problem. Output has problem + generation (proof text)."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_solve_for_quality.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/solved.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/proof/solve-proof-for-quality.yaml "
            f"++inference.tokens_to_generate=120000 "
            f"++inference.temperature=1.0 "
            f"++inference.top_p=0.95 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
            f"++skip_filled=True "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        dependent_jobs=4,
        **_deepseek_stage_kwargs(stage_config),
    )


def assess_problem_quality_with_solution(cluster, expname, run_after, stage_config, **kwargs):
    """Single-stage (combined): solve + judge in one call. Or stage 2 when using two-stage. Output: accepted/rejected."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_quality_assessment.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/accepted.jsonl "
        f"    {output_dir}/rejected.jsonl "
        f"    --stage problem_quality "
    )

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/proof/assess-problem-quality-with-solution-combined.yaml "
            f"++inference.tokens_to_generate=120000 "
            f"++inference.temperature=1.0 "
            f"++inference.top_p=0.95 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
            f"++skip_filled=True "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        dependent_jobs=4,
        **_deepseek_stage_kwargs(stage_config),
    )


stages_map = {
    "solve_proof_for_quality": solve_proof_for_quality,
    "assess_problem_quality_with_solution": assess_problem_quality_with_solution,
}


def get_available_configs(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    yaml_files = list(config_dir.glob("*.yaml"))
    return [f.stem for f in yaml_files if not f.name.startswith("template")]


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "configs"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(
        description="Solve-then-assess problem quality (DeepSeek, proof, single or 2-stage)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=available_configs,
        help="Config from rl-data-clean/configs/",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stages (default: all from config)",
    )
    args = parser.parse_args()

    config_path = config_dir / f"{args.config}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"Config {config_path} must define 'pipeline_stages'.")
    full_stage_sequence = config["pipeline_stages"]

    stages_to_run = args.stages.split(",") if args.stages else full_stage_sequence
    print(f"Stages: {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage '{stage}'. Available: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(f"Stage '{stage}' not in config sequence: {full_stage_sequence}")

    base_output_dir = config["base_output_dir"]
    suffix = config.get("suffix", args.config)
    cluster = config["cluster"]
    expname_base = config["expname"]

    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get("stages", {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)
        dep_stages = stage_config.get("dependencies", None)
        dependencies = None
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, d, suffix) for d in dep_stages]

        stage_func(
            cluster=cluster,
            expname=current_expname,
            run_after=dependencies,
            stage_config=stage_config,
        )

    print("\n--- Solve-then-assess pipeline finished. ---")
