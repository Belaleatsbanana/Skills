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
Non-Proof Problem Data Cleaning Pipeline - DeepSeek Version

This pipeline extracts and cleans high-quality non-proof math problems
from AOPS forum data for RL training using DeepSeek models.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, wrap_arguments


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def extract_problems(cluster, expname, run_after, stage_config, **kwargs):
    """Extracts potential problems from raw text data."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_problem_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-problems.jsonl "
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
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/common/extract-problems.yaml "
            f"++inference.top_p=0.95 "
            f"++inference.temperature=1.0 "
            f"++inference.tokens_to_generate=120000 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
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


def classify_problems(cluster, expname, run_after, stage_config, **kwargs):
    """
    Classifies extracted problems into different types using serial filtering.
    Each classifier outputs yes.jsonl and no.jsonl, with the next classifier
    processing the previous no.jsonl.

    Returns: dict with last_expname for dependency tracking
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    modes = stage_config["modes"]

    # Get server args from stage config with deepseek defaults
    server_args = stage_config.get(
        "server_args",
        "--ep-size 16 --dp 16 --enable-dp-attention --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 --mem-fraction-static=0.8",
    )

    current_run_after = run_after  # Initially from external dependency
    current_input_file = input_file
    last_mode_expname = None

    for mode in modes:
        mode_output_dir = f"{output_dir}/{mode}"
        mode_expname = f"{expname}-{mode}"

        postprocess_cmd = (
            f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_classification.py "
            f"    {mode_output_dir}/output.jsonl "
            f"    {mode_output_dir}/yes.jsonl "
            f"    {mode_output_dir}/no.jsonl "
            f"    --mode={mode}"
        )

        stage_kwargs = stage_config.get("stage_kwargs", {}).copy()
        stage_kwargs["server_args"] = server_args

        generate(
            ctx=wrap_arguments(
                f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/common/classify-if-{mode}.yaml "
                f"++inference.top_p=0.95 "
                f"++inference.temperature=1.0 "
                f"++inference.tokens_to_generate=120000 "
                f"++max_concurrent_requests=1024 "
                f"++inference.endpoint_type=chat "
                f"++chat_template_kwargs.thinking=true "
                f"{stage_config.get('inline_args', '')} "
            ),
            cluster=cluster,
            input_file=current_input_file,
            output_dir=mode_output_dir,
            postprocess_cmd=postprocess_cmd,
            expname=mode_expname,
            run_after=current_run_after,
            **stage_kwargs,
        )
        # Update for next iteration: run_after becomes a single expname (not list)
        current_run_after = mode_expname
        current_input_file = f"{mode_output_dir}/no.jsonl"
        last_mode_expname = mode_expname

    # Return the last expname so dependent stages wait for the complete chain
    return {"last_expname": last_mode_expname}


def assess_problem_quality(cluster, expname, run_after, stage_config, **kwargs):
    """Assess problem-only quality (before extracting answers)."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_quality_assessment.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/accepted.jsonl "
        f"    {output_dir}/rejected.jsonl "
        f"    --stage problem_only_quality "
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
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/non-proof/assess-problem-quality.yaml "
            f"++inference.top_p=0.95 "
            f"++inference.temperature=1.0 "
            f"++inference.tokens_to_generate=120000 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
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


def extract_answers(cluster, expname, run_after, stage_config, **kwargs):
    """Extract answers from forum discussions."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_answer_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-answers.jsonl "
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
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/non-proof/extract-answers.yaml "
            f"++inference.top_p=0.95 "
            f"++inference.temperature=1.0 "
            f"++inference.tokens_to_generate=120000 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
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


def extract_solution(cluster, expname, run_after, stage_config, **kwargs):
    """Extract and clean solution from forum discussions."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_solution_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-solutions.jsonl "
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
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/non-proof/extract-solution.yaml "
            f"++inference.top_p=0.95 "
            f"++inference.temperature=1.0 "
            f"++inference.tokens_to_generate=120000 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
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


def assess_problem_answer_quality(cluster, expname, run_after, stage_config, **kwargs):
    """Assess problem and answer quality (without solution)."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_quality_assessment.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/accepted.jsonl "
        f"    {output_dir}/rejected.jsonl "
        f"    --stage problem_answer_quality "
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
            f"++server.enable_soft_fail=True "
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


def assess_complete_solution_quality(cluster, expname, run_after, stage_config, **kwargs):
    """Assess complete problem-answer-solution quality."""
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
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/non-proof/assess-complete-solution-quality.yaml "
            f"++inference.top_p=0.95 "
            f"++inference.temperature=1.0 "
            f"++inference.tokens_to_generate=120000 "
            f"++max_concurrent_requests=1024 "
            f"++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true "
            f"++server.enable_soft_fail=True "
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
    "extract_problems": extract_problems,
    "classify_problems": classify_problems,
    "assess_problem_quality": assess_problem_quality,
    "extract_answers": extract_answers,
    "assess_problem_answer_quality": assess_problem_answer_quality,
    "extract_solution": extract_solution,
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

    parser = argparse.ArgumentParser(description="Non-Proof Problem Data Cleaning Pipeline - DeepSeek")
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
    last_expname_map = {}  # Track last expname for multi-sub-task stages

    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get("stages", {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)

        # Determine dependencies
        dependencies_list = stage_config.get("dependencies", [])
        if dependencies_list:
            # Use last_expname if available (for multi-sub-task stages like classify_problems)
            run_after = []
            for dep in dependencies_list:
                dep_expname = get_stage_expname(expname_base, dep, suffix)
                # If this dependency has a "last" expname (multi-sub-tasks), use that
                if dep in last_expname_map:
                    run_after.append(last_expname_map[dep])
                else:
                    run_after.append(dep_expname)
        else:
            run_after = None

        result = stage_func(
            cluster=cluster,
            expname=current_expname,
            run_after=run_after,
            stage_config=stage_config,
        )

        # If stage returns a "last" expname, track it
        if result and isinstance(result, dict) and "last_expname" in result:
            last_expname_map[stage] = result["last_expname"]

    print("\n=== Pipeline execution completed! ===")
