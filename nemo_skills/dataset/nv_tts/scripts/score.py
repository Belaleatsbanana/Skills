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

"""Scoring and aggregation functions for TTS evaluation."""

import argparse
import json
import os
import tempfile

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import evaluate


def run_scoring(
    results_dir: str,
    sv_model: str = "titanet",
    asr_model_name: str = "nvidia/parakeet-tdt-1.1b",
    language: str = "en",
    with_utmosv2: bool = False,
) -> None:
    """Run NeMo scoring on all benchmarks in results_dir."""
    benchmarks_dir = os.path.join(results_dir, "eval-results")
    if not os.path.exists(benchmarks_dir):
        benchmarks_dir = results_dir

    scoring_cfg = {
        "sv_model": sv_model,
        "asr_model_name": asr_model_name,
        "language": language,
        "with_utmosv2": with_utmosv2,
    }

    for benchmark in os.listdir(benchmarks_dir):
        benchmark_dir = os.path.join(benchmarks_dir, benchmark)
        if not os.path.isdir(benchmark_dir):
            continue

        output_jsonl = os.path.join(benchmark_dir, "output.jsonl")
        if not os.path.exists(output_jsonl):
            continue

        print(f"\nScoring: {benchmark}")
        metrics = score_benchmark(output_jsonl, scoring_cfg)

        # Save metrics.json
        metrics_path = os.path.join(benchmark_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {metrics_path}")
        print(f"  CER: {metrics.get('cer_cumulative', 'N/A'):.4f}")
        print(f"  WER: {metrics.get('wer_cumulative', 'N/A'):.4f}")
        if "utmosv2_avg" in metrics:
            print(f"  UTMOSv2: {metrics.get('utmosv2_avg', 'N/A'):.4f}")


def score_benchmark(output_jsonl: str, scoring_cfg: dict) -> dict:
    """Score a single benchmark."""
    # Parse output.jsonl
    entries = []
    records = []
    with open(output_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)

            # Extract manifest from user message
            manifest_entry = None
            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    manifest_entry = json.loads(content) if isinstance(content, str) else content
                    break

            audio_path = record.get("audio", {}).get("path")
            if audio_path and manifest_entry:
                entries.append((manifest_entry, audio_path))

    if not entries:
        return {}

    # Create temp dir with manifest and symlinks
    with tempfile.TemporaryDirectory(prefix="tts_scoring_") as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        gen_audio_dir = os.path.join(tmp_dir, "generated")
        os.makedirs(gen_audio_dir)

        with open(manifest_path, "w") as f:
            for i, (manifest_entry, audio_path) in enumerate(entries):
                f.write(json.dumps(manifest_entry) + "\n")
                dst = os.path.join(gen_audio_dir, f"predicted_audio_{i}.wav")
                if os.path.exists(audio_path):
                    os.symlink(audio_path, dst)

        avg_metrics, filewise_metrics = evaluate(
            manifest_path=manifest_path,
            audio_dir=None,
            generated_audio_dir=gen_audio_dir,
            language=scoring_cfg.get("language", "en"),
            sv_model_type=scoring_cfg.get("sv_model", "titanet"),
            asr_model_name=scoring_cfg.get("asr_model_name", "nvidia/parakeet-tdt-1.1b"),
            with_utmosv2=scoring_cfg.get("with_utmosv2", False),
        )

        # Save output_with_metrics.jsonl
        output_with_metrics_path = output_jsonl.replace("output.jsonl", "output_with_metrics.jsonl")
        with open(output_with_metrics_path, "w") as f:
            for i, record in enumerate(records):
                if i < len(filewise_metrics):
                    record["metrics"] = filewise_metrics[i]
                f.write(json.dumps(record) + "\n")
        print(f"Saved: {output_with_metrics_path}")

        return avg_metrics


def run_aggregation(results_dir: str) -> None:
    """Print summary of all metrics."""
    benchmarks_dir = os.path.join(results_dir, "eval-results")
    if not os.path.exists(benchmarks_dir):
        benchmarks_dir = results_dir

    print("\nAggregated Results:")
    for benchmark in sorted(os.listdir(benchmarks_dir)):
        metrics_path = os.path.join(benchmarks_dir, benchmark, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"  {benchmark}:")
            print(f"    CER: {metrics.get('cer_cumulative', 'N/A'):.4f}")
            print(f"    WER: {metrics.get('wer_cumulative', 'N/A'):.4f}")
            if "utmosv2_avg" in metrics:
                print(f"    UTMOSv2: {metrics.get('utmosv2_avg', 'N/A'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Scoring")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--sv_model", default="titanet")
    parser.add_argument("--asr_model_name", default="nvidia/parakeet-tdt-1.1b")
    parser.add_argument("--language", default="en")
    parser.add_argument("--with_utmosv2", action="store_true")
    parser.add_argument("--aggregation_only", action="store_true")
    args = parser.parse_args()

    if args.aggregation_only:
        run_aggregation(args.results_dir)
    else:
        run_scoring(
            args.results_dir,
            sv_model=args.sv_model,
            asr_model_name=args.asr_model_name,
            language=args.language,
            with_utmosv2=args.with_utmosv2,
        )
