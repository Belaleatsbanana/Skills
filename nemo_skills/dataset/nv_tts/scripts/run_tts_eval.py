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
TTS Pipeline: Generation -> Scoring -> Aggregation

Usage:
    python run_tts_eval.py --config config.yaml                           # all stages (fire-and-forget)
    python run_tts_eval.py --config config.yaml --stage scoring           # scoring only
    python run_tts_eval.py --config config.yaml --stage download          # download results locally
    python run_tts_eval.py --config config.yaml --stage download --wait   # poll until done, then download

All three stages (generation, scoring, aggregation of scoring results) are submitted as
dependent Slurm jobs. The aggregation job runs after all scoring jobs
complete and generates tts_eval_report.md on the cluster.

Use --stage download to fetch the report and metrics locally after jobs finish.
Add --wait to poll the aggregation job and download automatically when it completes.
"""

import argparse
import os
import shutil
import time

import yaml

from nemo_skills.dataset.nv_tts.scripts.resolve_code_repo import resolve_code_repo
from nemo_skills.dataset.nv_tts.scripts.score import run_aggregation
from nemo_skills.pipeline.eval import eval as ns_eval
from nemo_skills.pipeline.run_cmd import run_cmd as ns_run_cmd
from nemo_skills.pipeline.utils.cluster import get_cluster_config, get_tunnel


def wait_for_experiment(expname: str, poll_interval: int = 60):
    """Poll a nemo-run experiment until all its tasks reach a terminal state.

    Returns True if all tasks succeeded, False otherwise.
    """
    import nemo_run as run
    from torchx.specs.api import AppState

    terminal = {AppState.SUCCEEDED, AppState.FAILED, AppState.CANCELLED}

    print(f"Waiting for experiment '{expname}' to complete (polling every {poll_interval}s)...")
    while True:
        try:
            with run.Experiment.from_title(expname) as exp:
                status_dict = exp.status(return_dict=True)
        except (FileNotFoundError, AssertionError):
            print(f"  Experiment '{expname}' not found yet, retrying...")
            time.sleep(poll_interval)
            continue

        if not status_dict:
            print(f"  No tasks found in '{expname}', retrying...")
            time.sleep(poll_interval)
            continue

        all_terminal = True
        any_failed = False
        for task_name, info in status_dict.items():
            state = info["status"]
            if state in terminal:
                if state != AppState.SUCCEEDED:
                    any_failed = True
            else:
                all_terminal = False

        if all_terminal:
            if any_failed:
                print(f"  Experiment '{expname}' finished with failures.")
                return False
            print(f"  Experiment '{expname}' completed successfully.")
            return True

        states = ", ".join(f"{k}: {v['status'].name}" for k, v in status_dict.items())
        print(f"  [{time.strftime('%H:%M:%S')}] {states}")
        time.sleep(poll_interval)


class MockContext:
    """Mock typer.Context for programmatic calls."""

    def __init__(self, extra_args=None):
        self.args = extra_args or []


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_generation(cfg: dict, expname: str, output_dir: str, code_path: str = ""):
    """Run generation stage using ns eval, returns experiment object."""
    gen = cfg["generation"]

    # Add code_path to server_args (used for PYTHONPATH in the generation server)
    server_args = gen["server_args"]
    if code_path:
        server_args += f" --code_path {code_path}"

    # Parse extra_args for the context
    extra_args = gen.get("extra_args", "").split() if gen.get("extra_args") else []
    ctx = MockContext(extra_args)

    # Call eval programmatically
    return ns_eval(
        ctx=ctx,
        cluster=cfg["cluster"],
        output_dir=output_dir,
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
        choices=["all", "generation", "scoring", "download"],
        default="all",
        help="Stage to run. 'all' runs generation+scoring+aggregation on the cluster. "
        "'download' fetches results from the cluster after jobs complete.",
    )
    parser.add_argument("--expname", default="tts_eval", help="Base experiment name for job tracking")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll for job completion before running the download stage. "
        "Only meaningful with --stage download.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status polls when --wait is used (default: 60)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    scoring = cfg.get("scoring", {})
    hf_token = os.environ.get("HF_TOKEN", "")
    output_dir = os.path.join(cfg["output_dir"], args.expname)

    # Resolve code_path: explicit path takes precedence, then repo+commit, then legacy fields
    code_path = cfg.get("code_path") or cfg.get("nemo_code_path", "")
    if not code_path and cfg.get("code_repo"):
        cluster_config = get_cluster_config(cfg["cluster"])
        tunnel = get_tunnel(cluster_config)
        code_path = resolve_code_repo(
            cfg["code_repo"], cfg["code_commit"], output_dir, tunnel,
        )
        print(f"  Resolved code path: {code_path}")

    gen_exp_name = None

    # Stage 1: Generation
    if args.stage in ("all", "generation"):
        print("\n" + "=" * 60)
        print("Stage 1: GENERATION")
        print("=" * 60)
        gen_exp = run_generation(cfg, args.expname, output_dir, code_path=code_path)
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
        scoring_exp_names = []

        for benchmark in benchmarks:
            benchmark = benchmark.strip()
            # Benchmark dir in eval-results keeps dot notation (nv_tts.libritts_seen)
            benchmark_dir = benchmark
            # Short name for job (e.g. libritts_seen from nv_tts.libritts_seen)
            short_name = benchmark.split(".")[-1]

            # Base scoring command arguments (shared between chunked and non-chunked)
            base_scoring_args = (
                f"HF_TOKEN={hf_token} "
                f"PYTHONPATH={code_path}:$PYTHONPATH "
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
                scoring_exp_names.append(scoring_expname)
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

                scoring_expname = f"{args.expname}_score_{short_name}"
                scoring_exp_names.append(scoring_expname)

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
                    expname=scoring_expname,
                    log_dir=f"{output_dir}/eval-logs",
                )

        # Stage 3: Aggregation (submitted as dependent Slurm job)
        # Runs after all scoring jobs complete, generates the markdown report on the cluster.
        if scoring_exp_names:
            print("\n" + "=" * 60)
            print("Stage 3: AGGREGATION (dependent Slurm job)")
            print("=" * 60)

            agg_cmd = (
                f"python -m nemo_skills.dataset.nv_tts.scripts.score "
                f"--results_dir {output_dir} --aggregation_only"
            )
            agg_expname = f"{args.expname}_aggregate"
            print(f"  Submitting aggregation job (depends on: {', '.join(scoring_exp_names)})")

            ns_run_cmd(
                ctx=MockContext(),
                cluster=cfg["cluster"],
                container=cfg["container"],
                partition=cfg["partition"],
                num_gpus=1,
                mount_paths=cfg["mount_paths"],
                command=agg_cmd,
                run_after=scoring_exp_names,
                expname=agg_expname,
                log_dir=f"{output_dir}/eval-logs",
            )

            # Print the download command so the user can easily grab it later
            download_cmd = f"python {__file__} --config {args.config} --stage download"
            if args.expname != "tts_eval":
                download_cmd += f" --expname {args.expname}"
            wait_cmd = download_cmd + " --wait"
            print(f"\nTo auto-wait for jobs and download results:\n\n  {wait_cmd}\n")
            print(f"Or, if jobs are already done:\n\n  {download_cmd}\n")

    # Wait for jobs to finish before downloading (if --wait is set)
    if args.stage == "download" and args.wait:
        agg_expname = f"{args.expname}_aggregate"
        ok = wait_for_experiment(agg_expname, poll_interval=args.poll_interval)
        if not ok:
            print("WARNING: aggregation job did not succeed; downloading whatever is available.")

    # Download results from cluster (for local inspection after jobs complete)
    if args.stage == "download":
        print("\n" + "=" * 60)
        print("DOWNLOAD RESULTS")
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

            # Download the report generated by the aggregation Slurm job
            remote_report_path = os.path.join(output_dir, "tts_eval_report.md")
            local_report_path = os.path.join(local_results_dir, "tts_eval_report.md")
            check = tunnel.run(f"test -f {remote_report_path}", hide=True, warn=True)
            if check.return_code == 0:
                print(f"  Downloading tts_eval_report.md...")
                tunnel.get(remote_report_path, local_report_path)
            else:
                # Aggregation job may not have run yet; generate report locally from downloaded metrics
                print("  Report not found on cluster, generating locally from downloaded metrics...")
                run_aggregation(local_results_dir)

            # Copy report to a convenient location in the working directory
            report_path = os.path.join(local_results_dir, "tts_eval_report.md")
            if os.path.exists(report_path):
                convenient_path = os.path.join(os.getcwd(), f"tts_eval_report_{args.expname}.md")
                shutil.copy2(report_path, convenient_path)
                print(f"\nReport saved to: {convenient_path}")
                print(f"Full results downloaded to: {local_results_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
