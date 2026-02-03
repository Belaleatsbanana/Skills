#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

"""
Emergent TTS Pipeline: Generation -> Scoring (-> Aggregation)

This mirrors `nemo_skills/dataset/nv_tts/scripts/run_tts_eval.py` but uses
EmergentTTS-Eval scoring logic.
"""

import argparse
import os
from pathlib import Path

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
    gen = cfg["generation"]
    # Mirror nv_tts behavior: allow injecting a NeMo source checkout into PYTHONPATH
    # for the unified server (MagpieTTS inference code lives in NeMo).
    server_args = gen["server_args"]
    generation_code_path = cfg.get("generation_code_path") or cfg.get("nemo_code_path")
    if generation_code_path:
        server_args += f" --code_path {generation_code_path}"

    extra_args = gen.get("extra_args", "").split() if gen.get("extra_args") else []
    ctx = MockContext(extra_args)
    return ns_eval(
        ctx=ctx,
        cluster=cfg["cluster"],
        output_dir=cfg["output_dir"],
        benchmarks=gen["benchmarks"],
        model=gen["model"],
        server_type=gen["server_type"],
        server_gpus=gen["server_gpus"],
        # Local executor doesn't require explicit container/mount_paths in the run YAML.
        # For slurm clusters these are required and should be present in the config.
        server_container=cfg.get("container", ""),
        mount_paths=cfg.get("mount_paths", ""),
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
    parser = argparse.ArgumentParser(description="Emergent TTS Eval Pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "generation", "scoring", "aggregation"],
        default="all",
    )
    parser.add_argument("--expname", default="emergent_tts_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scoring = cfg.get("scoring", {})
    output_dir = cfg["output_dir"]

    gen_exp_name = None

    if args.stage in ("all", "generation"):
        print("\n" + "=" * 60)
        print("Stage 1: GENERATION")
        print("=" * 60)
        gen_exp = run_generation(cfg, args.expname)
        gen_exp_name = args.expname
        print(f"Generation submitted: {gen_exp}")

    if args.stage in ("all", "scoring"):
        print("\n" + "=" * 60)
        print("Stage 2: SCORING (EmergentTTS-Eval)")
        print("=" * 60)

        benchmarks = cfg["generation"]["benchmarks"].split(",")
        run_after = [gen_exp_name] if args.stage == "all" and gen_exp_name else None

        scoring_code_path = scoring.get("scoring_code_path", "")
        emergent_data_dir = scoring.get("emergent_data_dir", "")
        install_cmd = scoring.get("installation_command")
        scoring_container = scoring.get("container") or "nemo-skills"
        emergent_data_base_dir = str(Path(emergent_data_dir).parent) if emergent_data_dir else ""

        # Required by Emergent's judge clients
        judger_api_key = (
            os.environ.get("JUDGER_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        if not judger_api_key:
            print("Warning: JUDGER_API_KEY/NVIDIA_API_KEY/OPENAI_API_KEY not set; win_rate judging may fail.")

        for benchmark in benchmarks:
            benchmark = benchmark.strip()
            short_name = benchmark.split(".")[-1]
            score_cmd = (
                (f"cd {emergent_data_base_dir} && " if emergent_data_base_dir else "")
                + f"JUDGER_API_KEY={judger_api_key} "
                + f"PYTHONPATH={scoring_code_path}:$PYTHONPATH "
                + "python -m nemo_skills.dataset.emergent_tts.scripts.score "
                + f"--results_dir {output_dir} "
                + f"--benchmark {benchmark} "
                + f"--emergent_data_dir {emergent_data_dir} "
                + f"--judge_model {scoring.get('judge_model', 'gcp/google/gemini-2.5-pro')} "
                + f"--judger_base_url {scoring.get('judger_base_url', 'https://inference-api.nvidia.com/v1/chat/completions')} "
                + f"--num_threads {int(scoring.get('num_threads', 8))} "
                + f"--evaluate_function {scoring.get('evaluate_function', 'win_rate')}"
            )
            if scoring.get("strong_prompting"):
                score_cmd += " --strong_prompting"

            ns_run_cmd(
                ctx=MockContext(),
                cluster=cfg["cluster"],
                container=scoring_container,
                partition=cfg["partition"],
                num_gpus=int(scoring.get("gpus", 1)),
                mount_paths=cfg["mount_paths"],
                command=score_cmd,
                installation_command=install_cmd,
                run_after=run_after,
                # Ensure we ship the current repo state for scoring jobs.
                # (Otherwise nemo_run may reuse an older code snapshot and miss fixes.)
                reuse_code=False,
                expname=f"{args.expname}_score_{short_name}",
                log_dir=f"{output_dir}/eval-logs",
            )

    if args.stage == "aggregation":
        print("\n" + "=" * 60)
        print("Stage 3: AGGREGATION")
        print("=" * 60)
        agg_cmd = f"python -m nemo_skills.dataset.emergent_tts.scripts.score --results_dir {output_dir} --aggregation_only"
        ns_run_cmd(
            ctx=MockContext(),
            cluster=cfg["cluster"],
            container=cfg["container"],
            partition=cfg["partition"],
            num_gpus=0,
            mount_paths=cfg["mount_paths"],
            command=agg_cmd,
            reuse_code=False,
            expname=f"{args.expname}_agg",
            log_dir=f"{output_dir}/eval-logs",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()

