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

    # Add nemo_code_path to server_args
    server_args = gen["server_args"]
    if cfg.get("nemo_code_path"):
        server_args += f" --code_path {cfg['nemo_code_path']}"

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
    nemo_path = cfg["nemo_code_path"]
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

    # Stage 2: Scoring (one job per benchmark, depends on generation)
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

        for benchmark in benchmarks:
            benchmark = benchmark.strip()
            # Benchmark dir in eval-results keeps dot notation (nv_tts.libritts_seen)
            benchmark_dir = benchmark

            scoring_cmd = (
                f"HF_TOKEN={hf_token} "
                f"PYTHONPATH={nemo_path}:$PYTHONPATH "
                f"python -m nemo_skills.dataset.nv_tts.scripts.score "
                f"--results_dir {output_dir} "
                f"--benchmark {benchmark_dir} "
                f"--sv_model {scoring.get('sv_model', 'titanet')} "
                f"--asr_model_name {scoring.get('asr_model_name', 'nvidia/parakeet-tdt-1.1b')} "
                f"--language {scoring.get('language', 'en')}"
            )
            if scoring.get("with_utmosv2"):
                scoring_cmd += " --with_utmosv2"

            # Short name for job (e.g. libritts_seen from nv_tts.libritts_seen)
            short_name = benchmark.split(".")[-1]
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
        agg_cmd = f"python -m nemo_skills.dataset.nv_tts.scripts.score --results_dir {output_dir} --aggregation_only"
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
