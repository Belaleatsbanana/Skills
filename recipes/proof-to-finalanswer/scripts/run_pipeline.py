#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

GEN_STAGE = "run_generation"
BUILD_STAGE = "build_dataset"
ALL_STAGES = [GEN_STAGE, BUILD_STAGE]

DEFAULT_PROMPT_ROOT = "/nemo_run/code/recipes/proof-to-finalanswer/prompts"
DEFAULT_INLINE_ARGS = (
    "++inference.tokens_to_generate=110000 "
    "++inference.temperature=1.0 "
    "++inference.top_p=0.95"
)
DEFAULT_SERVER_ARGS = "--context-length 128000 --ep-size 16"


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def run_generation_stage(args, run_after: list[str] | None):
    expname = f"{args.expname}-{GEN_STAGE.replace('_', '-')}"
    ctx_args = (
        f"{args.inline_args} "
        f"++model_name={args.model} "
        f"++script_program_path=/nemo_run/code/recipes/proof-to-finalanswer/scripts/conversion_generation.py "
        f"++script_config.generate_solution_prompt_path={args.generate_solution_prompt_path} "
        f"++script_config.proof_transform_prompt_path={args.proof_transform_prompt_path} "
        f"++script_config.solution_comparison_prompt_path={args.solution_comparison_prompt_path} "
        f"++script_config.proof_problem_key={args.proof_problem_key} "
        f"++script_config.allow_provided_proof_solution={str(args.allow_provided_proof_solution).lower()} "
        f"++script_config.provided_proof_solution_keys={args.provided_proof_solution_keys} "
        f"++script_config.proof_solution_attempts={args.proof_solution_attempts} "
        f"++script_config.final_answer_solution_attempts={args.final_answer_solution_attempts} "
        f"++script_config.transform_attempts={args.transform_attempts} "
        f"++script_config.comparison_attempts={args.comparison_attempts} "
        f"++script_config.required_equivalence_score={args.required_equivalence_score} "
        f"++script_config.keep_attempt_generations={str(args.keep_attempt_generations).lower()} "
        f"++max_concurrent_requests={args.max_concurrent_requests} "
        f"++enable_litellm_cache=True "
    )

    generate(
        ctx=wrap_arguments(ctx_args),
        generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
        input_file=args.input_file,
        output_dir=args.output_dir,
        model=args.model,
        cluster=args.cluster,
        expname=expname,
        num_chunks=args.num_chunks,
        num_random_seeds=args.num_random_seeds,
        dependent_jobs=args.dependent_jobs,
        server_type=args.server_type,
        server_gpus=args.server_gpus,
        server_nodes=args.server_nodes,
        server_args=args.server_args,
        run_after=run_after,
        exclusive=True,
        partition="interactive",
    )
    return expname


def run_build_stage(args, run_after: list[str] | None):
    expname = f"{args.expname}-{BUILD_STAGE.replace('_', '-')}"
    cmd = (
        f"python /nemo_run/code/recipes/proof-to-finalanswer/scripts/build_dataset.py "
        f"--input_path {args.output_dir} "
        f"--output_file {args.build_output_file} "
        f"--required_equivalence_score {args.required_equivalence_score} "
        f"--dedup_key {args.dedup_key} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=args.cluster,
        expname=expname,
        run_after=run_after,
    )
    return expname


def main():
    cluster = "dfw"
    model = "/hf_models/DeepSeek-V3.2-Speciale"
    input = "ultra_proof_problems_subset"
    expname = f"transform_{input}"
    input_file = f"/workspace/finalanswer/data/{input}.jsonl"
    output_dir = f"/workspace/finalanswer/{expname}"
    server_gpus = 8
    server_nodes = 2
    num_chunks = 1
    proof_attempts = 8
    finalanswer_attempts = 8
    transform_attempts = 1
    comparison_attempts = 1
    max_concurrent_requests = 256

    parser = argparse.ArgumentParser(description="Proof-to-finalanswer pipeline runner.")
    parser.add_argument(
        "--stages",
        type=str,
        required=True,
        help=f"Comma-separated stages to run. Available: {', '.join(ALL_STAGES)}",
    )
    parser.add_argument("--cluster", type=str, default=cluster, help="Cluster config name.")
    parser.add_argument("--expname", type=str, default=expname)
    parser.add_argument("--run_after", type=str, default="", help="Comma-separated dependency experiment names.")

    parser.add_argument("--input_file", type=str, default=input_file, help="Input jsonl file with proof problems.")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Output directory for generation outputs.")
    parser.add_argument("--build_output_file", type=str, default=None, help="Output jsonl path for final dataset.")
    parser.add_argument("--dedup_key", type=str, default="problem")

    parser.add_argument("--model", type=str, default=model, help="Model path/name.")
    parser.add_argument("--server_type", type=str, default="sglang")
    parser.add_argument("--server_gpus", type=int, default=server_gpus)
    parser.add_argument("--server_nodes", type=int, default=server_nodes)
    parser.add_argument("--server_args", type=str, default=DEFAULT_SERVER_ARGS)
    parser.add_argument("--num_chunks", type=int, default=num_chunks)
    parser.add_argument("--num_random_seeds", type=int, default=1)
    parser.add_argument("--dependent_jobs", type=int, default=0)
    parser.add_argument("--inline_args", type=str, default=DEFAULT_INLINE_ARGS)
    parser.add_argument("--max_concurrent_requests", type=int, default=max_concurrent_requests)

    parser.add_argument(
        "--generate_solution_prompt_path",
        type=str,
        default=f"{DEFAULT_PROMPT_ROOT}/generate-solution.yaml",
    )
    parser.add_argument(
        "--proof_transform_prompt_path",
        type=str,
        default=f"{DEFAULT_PROMPT_ROOT}/proof-transform.yaml",
    )
    parser.add_argument(
        "--solution_comparison_prompt_path",
        type=str,
        default=f"{DEFAULT_PROMPT_ROOT}/solution-comparison.yaml",
    )

    parser.add_argument("--proof_problem_key", type=str, default="problem")
    parser.add_argument(
        "--allow_provided_proof_solution",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--provided_proof_solution_keys",
        type=str,
        default=None,
    )
    parser.add_argument("--proof_solution_attempts", type=int, default=proof_attempts)
    parser.add_argument("--final_answer_solution_attempts", type=int, default=finalanswer_attempts)
    parser.add_argument("--transform_attempts", type=int, default=transform_attempts)
    parser.add_argument("--comparison_attempts", type=int, default=comparison_attempts)
    parser.add_argument("--required_equivalence_score", type=float, default=1.0)
    parser.add_argument("--keep_attempt_generations", action="store_true")

    args = parser.parse_args()
    if args.build_output_file is None:
        args.build_output_file = f"{args.output_dir}/final_answer_dataset.jsonl"

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    for stage in stages:
        if stage not in ALL_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {ALL_STAGES}")

    run_after = _parse_csv_list(args.run_after)
    gen_expname = None

    if GEN_STAGE in stages:
        if args.input_file is None:
            raise ValueError("--input_file is required when running run_generation stage.")
        if args.model is None:
            raise ValueError("--model is required when running run_generation stage.")
        gen_expname = run_generation_stage(args, run_after=run_after)

    if BUILD_STAGE in stages:
        build_dependencies = run_after
        if gen_expname is not None:
            build_dependencies = [gen_expname]
        run_build_stage(args, run_after=build_dependencies)


if __name__ == "__main__":
    main()
