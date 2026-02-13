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

The aggregation stage downloads metrics.json and output_with_metrics.jsonl
from the cluster and runs report generation locally. This allows the
aggregation code to use any Python libraries available in the local
environment (e.g. statistics, scipy) without being constrained by what's
installed on the cluster login node or inside the container.
"""

import argparse
import os
import shutil

import yaml

from nemo_skills.dataset.nv_tts.scripts.score import run_aggregation
from nemo_skills.pipeline.eval import eval as ns_eval
from nemo_skills.pipeline.run_cmd import run_cmd as ns_run_cmd
from nemo_skills.pipeline.utils.cluster import get_cluster_config, get_tunnel


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
            if scoring.get("codec_model_path"):
                base_scoring_args += f" --codec_model_path {scoring['codec_model_path']}"

            if scoring_num_chunks and scoring_num_chunks > 1:
                # Parallel chunked scoring on a single node:
                # One Slurm job with num_tasks=num_chunks. Each task gets a
                # different $SLURM_LOCALID (0..N-1) which becomes the chunk_id.
                # Each task pins itself to its own GPU via CUDA_VISIBLE_DEVICES
                # to avoid all processes landing on GPU 0 and hitting OOM.
                # Each chunk auto-attempts the merge after finishing (self-aggregation),
                # so no separate aggregation job is needed.
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
            else:
                # Non-chunked: original single-job scoring per benchmark
                scoring_cmd = base_scoring_args

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

    # Stage 3: Aggregation (runs locally after downloading results from cluster)
    # Downloads metrics.json and output_with_metrics.jsonl per benchmark,
    # then runs aggregation/report generation locally where we have full
    # control over the Python environment (e.g. stats libraries).
    if args.stage == "aggregation":
        print("\n" + "=" * 60)
        print("Stage 3: AGGREGATION (local)")
        print("=" * 60)

        cluster_config = get_cluster_config(cfg["cluster"])
        tunnel = get_tunnel(cluster_config)

        # Create local results directory mirroring remote structure
        local_results_dir = os.path.join(os.getcwd(), "tts_results", args.expname)
        local_eval_dir = os.path.join(local_results_dir, "eval-results")
        os.makedirs(local_eval_dir, exist_ok=True)

        # Discover benchmarks on the remote cluster
        remote_eval_dir = os.path.join(output_dir, "eval-results")
        ls_result = tunnel.run(f"ls {remote_eval_dir}", hide=True, warn=True)
        if ls_result.return_code != 0:
            print(f"  ERROR: Could not list benchmarks at {remote_eval_dir}")
            print(ls_result.stderr)
        else:
            benchmarks = [b.strip() for b in ls_result.stdout.strip().split("\n") if b.strip()]
            print(f"  Found {len(benchmarks)} benchmark(s): {', '.join(benchmarks)}")

            # Download metrics.json and output_with_metrics.jsonl for each benchmark
            for benchmark in benchmarks:
                remote_bench_dir = os.path.join(remote_eval_dir, benchmark)
                local_bench_dir = os.path.join(local_eval_dir, benchmark)
                os.makedirs(local_bench_dir, exist_ok=True)

                for filename in ["metrics.json", "output_with_metrics.jsonl"]:
                    remote_path = os.path.join(remote_bench_dir, filename)
                    local_path = os.path.join(local_bench_dir, filename)

                    # Check if the remote file exists before downloading
                    check = tunnel.run(f"test -f {remote_path}", hide=True, warn=True)
                    if check.return_code != 0:
                        print(f"  WARNING: {benchmark}/{filename} not found on cluster, skipping")
                        continue

                    print(f"  Downloading {benchmark}/{filename}...")
                    tunnel.get(remote_path, local_path)

            # Run aggregation locally on the downloaded files
            print("\n  Running aggregation locally...")
            run_aggregation(local_results_dir)

            # Copy report to a convenient location in the working directory
            report_path = os.path.join(local_results_dir, "tts_eval_report.md")
            if os.path.exists(report_path):
                local_report_path = os.path.join(os.getcwd(), f"tts_eval_report_{args.expname}.md")
                shutil.copy2(report_path, local_report_path)
                print(f"\nReport saved to: {local_report_path}")
                print(f"Full results downloaded to: {local_results_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
