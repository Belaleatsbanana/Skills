#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

"""Run EmergentTTS-Eval scoring on NeMo-Skills generated audio.

This script expects NeMo-Skills generation output layout:
  <results_dir>/eval-results/<benchmark>/output.jsonl

It will:
  1) Convert NeMo-Skills `output.jsonl` audio paths into Emergent layout
     (<benchmark>/emergent-tts-eval_output-audios/<unique_id_eval>.wav)
  2) Run Emergent scoring in fetch-audios mode (no re-generation)
  3) Write `metrics.json` in the benchmark folder for consistency with other evals
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _benchmarks_dir(results_dir: str) -> Path:
    p = Path(results_dir) / "eval-results"
    return p if p.exists() else Path(results_dir)


def _normalize_openai_base_url(url: str) -> str:
    # Some callers pass the full endpoint; OpenAI client expects base URL.
    suffix = "/v1/chat/completions"
    if url.endswith(suffix):
        return url[: -len("/chat/completions")]
    return url


class _NoopModelClient:
    """A minimal Emergent model_client for scoring-only runs."""

    def prepare_emergent_tts_sample(self, text_to_synthesize, category, strong_prompting, prompting_object, **kwargs):
        if strong_prompting:
            user_message = (
                prompting_object.USER_MESSAGE_STRONG_TEMPLATE.replace(
                    "{{{descriptions}}}", prompting_object.ALL_DESCRIPTIONS[category]
                ).replace("{{{text_to_synthesize}}}", text_to_synthesize)
            )
        else:
            user_message = prompting_object.USER_MESSAGE_DEFAULT_TEMPLATE.replace(
                "{{{text_to_synthesize}}}", text_to_synthesize
            )
        return prompting_object.SYSTEM_PROMPT_DEFAULT, user_message


def _convert(ns_output_jsonl: Path, out_dir: Path, overwrite: bool) -> None:
    from nemo_skills.dataset.emergent_tts.scripts.convert_ns_outputs_to_emergent import main as convert_main

    # Reuse converter as a library via argv.
    import sys

    argv = sys.argv
    try:
        sys.argv = [
            argv[0],
            "--ns_output_jsonl",
            str(ns_output_jsonl),
            "--out_dir",
            str(out_dir),
            "--mode",
            "symlink",
        ] + (["--overwrite"] if overwrite else [])
        convert_main()
    finally:
        sys.argv = argv


def _run_emergent_scoring(
    *,
    benchmark_dir: Path,
    emergent_data_base_path: Path,
    fetch_audios_from_path: Path,
    baseline_audios_path: Path,
    judge_model: str,
    judger_base_url: str,
    num_threads: int,
    depths_to_evaluate: str,
    categories_to_evaluate: str,
    evaluate_function: str,
    strong_prompting: bool,
):
    # Import from EmergentTTS-Eval-public (caller should add it to PYTHONPATH).
    import inference as emergent_inference  # type: ignore

    # Tell Emergent code where to find `emergent_tts_eval_data.jsonl` and `wv_mos.ckpt`.
    os.environ["EMERGENT_TTS_DATA_BASE_PATH"] = str(emergent_data_base_path)

    # EmergentTTS-Eval expects paths like "data/emergent_tts_eval_data.jsonl" relative
    # to its *data base directory* (repo root). We keep the dataset in a shared path:
    #   <...>/emergent_tts/data/{emergent_tts_eval_data.jsonl,wv_mos.ckpt,baseline_audios/}
    # So we temporarily `chdir` into the directory that contains the "data/" folder.
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(emergent_data_base_path.parent))
        emergent_inference.eval_api_closed_model(
            model_client=_NoopModelClient(),
            accelerator=None,
            depths_to_evaluate=depths_to_evaluate,
            categories_to_evaluate=categories_to_evaluate,
            seed=42,
            output_dir=str(benchmark_dir),
            num_samples=None,
            baseline_audios_path=str(baseline_audios_path),
            fetch_audios_from_path=str(fetch_audios_from_path),
            judge_model=judge_model,
            temperature=0.0,
            evaluate_function=evaluate_function,
            strong_prompting=strong_prompting,
            judger_base_url=_normalize_openai_base_url(judger_base_url) if judger_base_url else None,
            num_threads=num_threads,
            model_name="nemo-skills-generated",
        )
    finally:
        os.chdir(prev_cwd)


def run_scoring(
    *,
    results_dir: str,
    benchmark: str | None,
    emergent_data_dir: str,
    judge_model: str,
    judger_base_url: str,
    num_threads: int,
    depths_to_evaluate: str,
    categories_to_evaluate: str,
    evaluate_function: str,
    strong_prompting: bool,
    overwrite_converted: bool,
):
    bdir = _benchmarks_dir(results_dir)
    emergent_data_dir_p = Path(emergent_data_dir)
    emergent_base = emergent_data_dir_p  # expects emergent_tts_eval_data.jsonl and wv_mos.ckpt here
    baseline_audios = emergent_data_dir_p / "baseline_audios"

    if benchmark:
        benches = [benchmark]
    else:
        benches = [p.name for p in bdir.iterdir() if p.is_dir()]

    for bench in sorted(benches):
        bench_dir = bdir / bench
        output_jsonl = bench_dir / "output.jsonl"
        if not output_jsonl.exists():
            print(f"Skipping {bench}: output.jsonl not found")
            continue

        # Emergent uses this naming convention for generated audio dir (see inference.py).
        converted_audio_dir = bench_dir / "emergent-tts-eval_output-audios"
        converted_audio_dir.mkdir(parents=True, exist_ok=True)
        _convert(output_jsonl, converted_audio_dir, overwrite=overwrite_converted)

        # Run Emergent scoring (writes emergent-tts-eval_* files into bench_dir)
        _run_emergent_scoring(
            benchmark_dir=bench_dir,
            emergent_data_base_path=emergent_base,
            fetch_audios_from_path=converted_audio_dir,
            baseline_audios_path=baseline_audios,
            judge_model=judge_model,
            judger_base_url=judger_base_url,
            num_threads=num_threads,
            depths_to_evaluate=depths_to_evaluate,
            categories_to_evaluate=categories_to_evaluate,
            evaluate_function=evaluate_function,
            strong_prompting=strong_prompting,
        )

        # Convert Emergent metrics file into `metrics.json` for NeMo-Skills conventions.
        # Emergent prefix matches inference.py defaults when strong_prompting=False and voice_to_use=None.
        emergent_metrics_path = bench_dir / "emergent-tts-eval_evaluation-metrics.json"
        if emergent_metrics_path.exists():
            with open(emergent_metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            with open(bench_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[{bench}] Saved: {bench_dir/'metrics.json'}")
        else:
            print(f"[{bench}] Warning: Emergent metrics file not found at {emergent_metrics_path}")


def run_aggregation(results_dir: str):
    bdir = _benchmarks_dir(results_dir)
    print("\nAggregated Results (EmergentTTS-Eval):")
    for benchmark in sorted([p.name for p in bdir.iterdir() if p.is_dir()]):
        metrics_path = bdir / benchmark / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        # Keep this minimal; Emergent metrics are keyed like eval/wer, eval/mos, eval/win_rate, etc.
        wer = metrics.get("eval/wer")
        mos = metrics.get("eval/mos")
        win = metrics.get("eval/win_rate")
        print(f"  {benchmark}:")
        if wer is not None:
            print(f"    WER: {wer:.4f}")
        if mos is not None:
            print(f"    MOS: {mos:.4f}")
        if win is not None:
            print(f"    Win-rate: {win:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmergentTTS-Eval scoring for NeMo-Skills outputs")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--benchmark", default=None, help="Score only this benchmark (e.g. emergent_tts.emergent)")
    parser.add_argument("--aggregation_only", action="store_true")

    parser.add_argument(
        "--emergent_data_dir",
        required=False,
        default=None,
        help="Path containing Emergent files: emergent_tts_eval_data.jsonl, wv_mos.ckpt, baseline_audios/",
    )
    parser.add_argument("--judge_model", default="gcp/google/gemini-2.5-pro")
    parser.add_argument("--judger_base_url", default="https://inference-api.nvidia.com/v1/chat/completions")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--depths_to_evaluate", default="0,1,2,3")
    parser.add_argument(
        "--categories_to_evaluate",
        default="Emotions,Paralinguistics,Syntactic Complexity,Foreign Words,Questions,Pronunciation",
    )
    parser.add_argument("--evaluate_function", default="win_rate")
    parser.add_argument("--strong_prompting", action="store_true")
    parser.add_argument("--overwrite_converted", action="store_true")
    args = parser.parse_args()

    if args.aggregation_only:
        run_aggregation(args.results_dir)
    else:
        emergent_data_dir = args.emergent_data_dir
        if emergent_data_dir is None:
            # Try to derive from NEMO_SKILLS_DATA_DIR (common in cluster configs).
            emergent_data_dir = os.environ.get("EMERGENT_TTS_DATA_BASE_PATH") or os.environ.get("NEMO_SKILLS_DATA_DIR")
            if emergent_data_dir:
                emergent_data_dir = str(Path(emergent_data_dir) / "emergent_tts" / "data")
        if emergent_data_dir is None:
            raise SystemExit("--emergent_data_dir is required (or set EMERGENT_TTS_DATA_BASE_PATH/NEMO_SKILLS_DATA_DIR)")

        run_scoring(
            results_dir=args.results_dir,
            benchmark=args.benchmark,
            emergent_data_dir=emergent_data_dir,
            judge_model=args.judge_model,
            judger_base_url=args.judger_base_url,
            num_threads=args.num_threads,
            depths_to_evaluate=args.depths_to_evaluate,
            categories_to_evaluate=args.categories_to_evaluate,
            evaluate_function=args.evaluate_function,
            strong_prompting=args.strong_prompting,
            overwrite_converted=args.overwrite_converted,
        )

