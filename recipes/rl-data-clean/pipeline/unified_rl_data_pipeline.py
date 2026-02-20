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
Unified RL data pipeline: extract -> filter (invalid/binary/mcq) -> dedup (retrieve_similar + check_contamination + cluster) -> solve+difficulty -> classify proof vs non-proof -> non-proof: extract answer.

Run from repo root:
  python recipes/rl-data-clean/pipeline/unified_rl_data_pipeline.py --config unified-rl-deepseek
  python recipes/rl-data-clean/pipeline/unified_rl_data_pipeline.py --config unified-rl-deepseek --stages solve_and_difficulty,classify_if_proof
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments


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


# ---------- 1. Extract problems ----------
def extract_problems(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_problem_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-problems.jsonl "
    )
    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/common/extract-problems.yaml "
            f"++inference.top_p=0.95 ++inference.temperature=1.0 ++inference.tokens_to_generate=16384 "
            f"++max_concurrent_requests=1024 ++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true ++server.enable_soft_fail=True ++skip_filled=True "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        dependent_jobs=2,
        **_deepseek_stage_kwargs(stage_config),
    )


# ---------- 2. Filter: invalid -> binary -> mcq (keep "no" each time) ----------
def filter_invalid_binary_mcq(cluster, expname, run_after, stage_config, **kwargs):
    """Run classify invalid, then binary, then mcq; each keeps no.jsonl for next. Final output: filtered.jsonl (last no.jsonl)."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    modes = stage_config.get("modes", ["invalid", "binary", "mcq"])
    current_run_after = run_after
    current_input = input_file
    for i, mode in enumerate(modes):
        mode_dir = f"{output_dir}/{mode}"
        mode_expname = f"{expname}-{mode}"
        postprocess_cmd = (
            f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_classification.py "
            f"    {mode_dir}/output.jsonl "
            f"    {mode_dir}/yes.jsonl "
            f"    {mode_dir}/no.jsonl "
            f"    --mode={mode}"
        )
        generate(
            ctx=wrap_arguments(
                f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/common/classify-if-{mode}.yaml "
                f"++inference.top_p=0.95 ++inference.temperature=1.0 ++inference.tokens_to_generate=16384 "
                f"++max_concurrent_requests=1024 ++inference.endpoint_type=chat "
                f"++chat_template_kwargs.thinking=true ++server.enable_soft_fail=True ++skip_filled=True "
                f"{stage_config.get('inline_args', '')} "
            ),
            cluster=cluster,
            input_file=current_input,
            output_dir=mode_dir,
            postprocess_cmd=postprocess_cmd,
            expname=mode_expname,
            run_after=current_run_after,
            dependent_jobs=2,
            **_deepseek_stage_kwargs(stage_config),
        )
        current_run_after = mode_expname
        current_input = f"{mode_dir}/no.jsonl"
    # Copy last no.jsonl to filtered.jsonl so next stage depends on this expname
    copy_cmd = f"cp {output_dir}/{modes[-1]}/no.jsonl {output_dir}/filtered.jsonl"
    run_cmd(
        ctx=wrap_arguments(copy_cmd),
        cluster=cluster,
        container=stage_config.get("container", "nemo-rl"),
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=current_run_after,
        num_nodes=1,
        num_gpus=0,
    )


# ---------- 3. Retrieve similar (self-vs-self, top_k=20) ----------
def retrieve_similar(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    retrieve_from = stage_config.get("retrieve_from", input_file)
    top_k = stage_config.get("top_k", 20)
    retrieve_key = stage_config.get("retrieve_key", "problem")
    out_file = f"{output_dir}/retrieved.jsonl"
    cmd = (
        f"python -m nemo_skills.inference.retrieve_similar "
        f"  ++retrieve_from='{retrieve_from}' "
        f"  ++compare_to={input_file} "
        f"  ++output_file={out_file} "
        f"  ++top_k={top_k} "
        f"  ++retrieve_key={retrieve_key} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        container=stage_config.get("container", "nemo-rl"),
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        installation_command="pip install sentence-transformers",
        num_nodes=1,
        num_gpus=stage_config.get("num_gpus", 0),
        **stage_config.get("run_cmd_kwargs", {}),
    )


# ---------- 4. Check contamination (LLM judges same/different for each of top_k) ----------
def check_contamination(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    generate(
        ctx=wrap_arguments(
            "++skip_filled=True ++inference.top_p=0.95 ++inference.temperature=1.0 "
            f"{stage_config.get('inline_args', '')} "
        ),
        generation_type="check_contamination",
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        dependent_jobs=2,
        **stage_config.get("stage_kwargs", {}),
    )


# ---------- 5. Cluster by contamination (Union-Find) -> dedup.jsonl ----------
def cluster_dedup(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    script = stage_config.get(
        "cluster_script",
        "/nemo_run/code/recipes/rl-data-clean/scripts/cluster_by_contamination.py",
    )
    dedup_file = f"{output_dir}/dedup.jsonl"
    cmd = f"python {script} {input_file} --out-dedup {dedup_file}"
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        container=stage_config.get("container", "nemo-rl"),
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_nodes=1,
        num_gpus=0,
    )


# ---------- 5b. Merge metadata back into dedup (source = filtered.jsonl full rows) ----------
def merge_dedup_metadata(cluster, expname, run_after, stage_config, **kwargs):
    """After cluster_dedup, attach full-row metadata from source (e.g. filtered.jsonl) to dedup.jsonl."""
    output_dir = stage_config["output_dir"]
    source_file = stage_config["source_file"]
    dedup_file = stage_config["dedup_file"]
    out_file = stage_config["output_file"]
    merge_script = stage_config.get(
        "merge_script",
        "/nemo_run/code/recipes/rl-data-clean/scripts/merge_dedup_metadata.py",
    )
    cmd = f"python {merge_script} --source {source_file} --dedup {dedup_file} --output {out_file}"
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        container=stage_config.get("container", "nemo-rl"),
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=run_after,
        num_nodes=1,
        num_gpus=0,
    )


# ---------- 6. Solve + difficulty (generic: one prompt for all problems) ----------
def solve_and_difficulty(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    prompt = stage_config.get(
        "prompt_config",
        "/nemo_run/code/recipes/rl-data-clean/prompts/common/solve-and-difficulty-combined.yaml",
    )
    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_quality_assessment.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/accepted.jsonl "
        f"    {output_dir}/rejected.jsonl "
        f"    --stage problem_quality "
    )
    generate(
        ctx=wrap_arguments(
            f"++prompt_config={prompt} "
            f"++inference.tokens_to_generate=120000 ++inference.temperature=1.0 ++inference.top_p=0.95 "
            f"++max_concurrent_requests=1024 ++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true ++server.enable_soft_fail=True ++skip_filled=True "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        dependent_jobs=2,
        **_deepseek_stage_kwargs(stage_config),
    )


# ---------- 7. Classify proof vs non-proof ----------
def classify_if_proof(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_classification.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/proof.jsonl "
        f"    {output_dir}/non_proof.jsonl "
        f"    --mode proof "
    )
    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/common/classify-if-proof.yaml "
            f"++inference.top_p=0.95 ++inference.temperature=1.0 ++inference.tokens_to_generate=16384 "
            f"++max_concurrent_requests=1024 ++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true ++server.enable_soft_fail=True ++skip_filled=True "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        dependent_jobs=2,
        **_deepseek_stage_kwargs(stage_config),
    )


# ---------- 8. Non-proof: extract expected/final answer (for RL) ----------
def extract_answer_non_proof(cluster, expname, run_after, stage_config, **kwargs):
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    postprocess_cmd = (
        f"python /nemo_run/code/recipes/rl-data-clean/scripts/postprocess_solution_extraction.py "
        f"    {output_dir}/output.jsonl "
        f"    {output_dir}/extracted-answers.jsonl "
    )
    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/rl-data-clean/prompts/common/extract-expected-answer.yaml "
            f"++inference.top_p=0.95 ++inference.temperature=1.0 ++inference.tokens_to_generate=8192 "
            f"++max_concurrent_requests=1024 ++inference.endpoint_type=chat "
            f"++chat_template_kwargs.thinking=true ++server.enable_soft_fail=True ++skip_filled=True "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        postprocess_cmd=postprocess_cmd,
        expname=expname,
        run_after=run_after,
        dependent_jobs=2,
        **_deepseek_stage_kwargs(stage_config),
    )


stages_map = {
    "extract_problems": extract_problems,
    "filter_invalid_binary_mcq": filter_invalid_binary_mcq,
    "retrieve_similar": retrieve_similar,
    "check_contamination": check_contamination,
    "cluster_dedup": cluster_dedup,
    "merge_dedup_metadata": merge_dedup_metadata,
    "solve_and_difficulty": solve_and_difficulty,
    "classify_if_proof": classify_if_proof,
    "extract_answer_non_proof": extract_answer_non_proof,
}


def get_available_configs(config_dir):
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    return [f.stem for f in config_dir.glob("*.yaml") if not f.name.startswith("template")]


if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "configs"
    available = get_available_configs(config_dir)
    parser = argparse.ArgumentParser(
        description="Unified RL data pipeline: extract -> filter -> dedup -> solve -> proof/non-proof"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config name without .yaml (e.g. unified-rl-deepseek)"
    )
    parser.add_argument("--stages", type=str, default=None, help="Comma-separated stages to run (default: all)")
    args = parser.parse_args()

    config_path = config_dir / f"{args.config}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Available: {available}")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    pipeline_stages = config.get("pipeline_stages", list(stages_map.keys()))
    stages_to_run = [s.strip() for s in args.stages.split(",")] if args.stages else pipeline_stages
    for s in stages_to_run:
        if s not in stages_map:
            raise ValueError(f"Unknown stage: {s}. Available: {list(stages_map.keys())}")

    base_output_dir = config["base_output_dir"]
    suffix = config.get("suffix", args.config)
    cluster = config["cluster"]
    expname_base = config["expname"]

    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_config = config.get("stages", {}).get(stage, {})
        stage_func = stages_map[stage]
        current_expname = get_stage_expname(expname_base, stage, suffix)
        deps = stage_config.get("dependencies")
        run_after = [get_stage_expname(expname_base, d, suffix) for d in deps] if deps else None
        stage_func(cluster=cluster, expname=current_expname, run_after=run_after, stage_config=stage_config)

    print("\n--- Unified pipeline finished. ---")
