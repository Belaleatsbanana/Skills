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
TTS Pipeline: Generation -> Scoring (-> Aggregation)

Usage:
    python run_tts_eval.py --config config.yaml
    python run_tts_eval.py --config config.yaml --stage scoring
    python run_tts_eval.py --config config.yaml --stage aggregation
"""

import argparse
import os

import yaml

from nemo_skills.pipeline.eval import eval as ns_eval
from nemo_skills.pipeline.run_cmd import run_cmd as ns_run_cmd


class MockContext:
    """Mock typer.Context for programmatic calls."""

    def __init__(self, extra_args=None):
        self.args = extra_args or []


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_generation(cfg: dict, expname: str):
    """Run generation stage using ns eval, returns experiment object."""
    gen = cfg["generation"]

    # Add generation_code_path to server_args
    server_args = gen["server_args"]
    if cfg.get("generation_code_path"):
        server_args += f" --code_path {cfg['generation_code_path']}"

    # Parse extra_args for the context
    extra_args = gen.get("extra_args", "").split() if gen.get("extra_args") else []
    ctx = MockContext(extra_args)

    # Call eval programmatically
    return ns_eval(
        ctx=ctx,
        cluster=cfg["cluster"],
        output_dir=cfg["output_dir"],
        benchmarks=gen["benchmarks"],
        model=gen["model"],
        server_type=gen["server_type"],
        server_gpus=gen["server_gpus"],
        server_container=cfg["container"],
        mount_paths=cfg["mount_paths"],
        server_entrypoint=gen["server_entrypoint"],
        server_args=server_args,
        data_dir=gen["data_dir"],
        num_chunks=gen["num_chunks"],
        gpus_per_node=gen.get("gpus_per_node", 1),
        partition=cfg["partition"],
        expname=expname,
        auto_summarize_results=False,
    )


def main():
    parser = argparse.ArgumentParser(description="TTS Pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "generation", "scoring", "aggregation"],
        default="all",
        help="Stage to run. 'all' runs generation+scoring (no aggregation)",
    )
    parser.add_argument("--expname", default="tts_eval", help="Base experiment name for job tracking")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scoring = cfg.get("scoring", {})
    hf_token = os.environ.get("HF_TOKEN", "")
    scoring_code_path = cfg.get("scoring_code_path", "")
    output_dir = cfg["output_dir"]

    gen_exp_name = None

    # Stage 1: Generation
    if args.stage in ("all", "generation"):
        print("\n" + "=" * 60)
        print("Stage 1: GENERATION")
        print("=" * 60)
        gen_exp = run_generation(cfg, args.expname)
        # Extract experiment name/id for dependency tracking
        gen_exp_name = args.expname  # The expname we passed to ns_eval
        print(f"Generation submitted: {gen_exp}")

    # Stage 2: Scoring (depends on generation)
    if args.stage in ("all", "scoring"):
        print("\n" + "=" * 60)
        print("Stage 2: SCORING")
        print("=" * 60)

        # Parse benchmarks list
        benchmarks = cfg["generation"]["benchmarks"].split(",")

        install_cmd = None
        if scoring.get("with_utmosv2"):
            install_cmd = "pip install git+https://github.com/sarulab-speech/UTMOSv2.git@v1.2.1"

        # When running both stages, scoring depends on generation experiment (by name)
        run_after = [gen_exp_name] if args.stage == "all" and gen_exp_name else None

        scoring_num_chunks = scoring.get("num_chunks")

        for benchmark in benchmarks:
            benchmark = benchmark.strip()
            # Benchmark dir in eval-results keeps dot notation (nv_tts.libritts_seen)
            benchmark_dir = benchmark
            # Short name for job (e.g. libritts_seen from nv_tts.libritts_seen)
            short_name = benchmark.split(".")[-1]

            # Base scoring command arguments (shared between chunked and non-chunked)
            base_scoring_args = (
                f"HF_TOKEN={hf_token} "
                f"PYTHONPATH={scoring_code_path}:$PYTHONPATH "
                f"python -m nemo_skills.dataset.nv_tts.scripts.score "
                f"--results_dir {output_dir} "
                f"--benchmark {benchmark_dir} "
                f"--sv_model {scoring.get('sv_model', 'titanet')} "
                f"--asr_model_name {scoring.get('asr_model_name', 'nvidia/parakeet-tdt-1.1b')} "
                f"--language {scoring.get('language', 'en')}"
            )
            if scoring.get("with_utmosv2"):
                base_scoring_args += " --with_utmosv2"

            if scoring_num_chunks and scoring_num_chunks > 1:
                # Parallel chunked scoring on a single node:
                # One Slurm job with num_tasks=num_chunks. Each task gets a
                # different $SLURM_LOCALID (0..N-1) which becomes the chunk_id.
                # Each task pins itself to its own GPU via CUDA_VISIBLE_DEVICES
                # to avoid all processes landing on GPU 0 and hitting OOM.
                chunk_cmd = (
                    f"export CUDA_VISIBLE_DEVICES=${{SLURM_LOCALID:-0}} && "
                    f"{base_scoring_args} "
                    f'--num_chunks {scoring_num_chunks} --chunk_id ${{SLURM_LOCALID:-0}}'
                )
                scoring_expname = f"{args.expname}_score_{short_name}"
                print(
                    f"  Submitting multi-instance scoring job for: {benchmark} "
                    f"({scoring_num_chunks} chunks on {scoring_num_chunks} GPUs)"
                )

                ns_run_cmd(
                    ctx=MockContext(),
                    cluster=cfg["cluster"],
                    container=cfg["container"],
                    partition=cfg["partition"],
                    num_gpus=scoring_num_chunks,
                    num_tasks=scoring_num_chunks,
                    mount_paths=cfg["mount_paths"],
                    command=chunk_cmd,
                    installation_command=install_cmd,
                    run_after=run_after,
                    expname=scoring_expname,
                    log_dir=f"{output_dir}/eval-logs",
                )

                # Aggregation job: merge chunks, recompute global metrics, compute FCD.
                # Depends on the scoring job above.
                agg_cmd = (
                    f"HF_TOKEN={hf_token} "
                    f"PYTHONPATH={scoring_code_path}:$PYTHONPATH "
                    f"python -m nemo_skills.dataset.nv_tts.scripts.score "
                    f"--results_dir {output_dir} "
                    f"--benchmark {benchmark_dir} "
                    f"--merge_scoring_chunks "
                    f"--num_chunks {scoring_num_chunks}"
                )
                if scoring.get("with_fcd"):
                    agg_cmd += " --with_fcd"
                    if scoring.get("codec_model_path"):
                        agg_cmd += f" --codec_model_path {scoring['codec_model_path']}"

                agg_expname = f"{args.expname}_score_{short_name}_agg"
                print(f"  Submitting scoring aggregation job for: {benchmark}")

                ns_run_cmd(
                    ctx=MockContext(),
                    cluster=cfg["cluster"],
                    container=cfg["container"],
                    partition=cfg["partition"],
                    num_gpus=1 if scoring.get("with_fcd") else 0,
                    mount_paths=cfg["mount_paths"],
                    command=agg_cmd,
                    run_after=[scoring_expname],
                    expname=agg_expname,
                    log_dir=f"{output_dir}/eval-logs",
                )
            else:
                # Non-chunked: original single-job scoring per benchmark
                scoring_cmd = base_scoring_args
                if scoring.get("with_fcd"):
                    scoring_cmd += " --with_fcd"
                    if scoring.get("codec_model_path"):
                        scoring_cmd += f" --codec_model_path {scoring['codec_model_path']}"

                print(f"  Submitting scoring job for: {benchmark}")

                ns_run_cmd(
                    ctx=MockContext(),
                    cluster=cfg["cluster"],
                    container=cfg["container"],
                    partition=cfg["partition"],
                    num_gpus=scoring.get("gpus", 1),
                    mount_paths=cfg["mount_paths"],
                    command=scoring_cmd,
                    installation_command=install_cmd,
                    run_after=run_after,
                    expname=f"{args.expname}_score_{short_name}",
                    log_dir=f"{output_dir}/eval-logs",
                )

    # Stage 3: Aggregation (only if explicitly requested)
    if args.stage == "aggregation":
        print("\n" + "=" * 60)
        print("Stage 3: AGGREGATION")
        print("=" * 60)
        # score.py imports NeMo at top level; container needs NeMo on PYTHONPATH
        agg_cmd = (
            f"PYTHONPATH={scoring_code_path}:$PYTHONPATH "
            f"python -m nemo_skills.dataset.nv_tts.scripts.score --results_dir {output_dir} --aggregation_only"
        )
        ns_run_cmd(
            ctx=MockContext(),
            cluster=cfg["cluster"],
            container=cfg["container"],
            partition=cfg["partition"],
            num_gpus=0,
            mount_paths=cfg["mount_paths"],
            command=agg_cmd,
            expname=f"{args.expname}_agg",
            log_dir=f"{output_dir}/eval-logs",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
