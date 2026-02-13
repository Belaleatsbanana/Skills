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

"""Scoring and aggregation functions for TTS evaluation.

Supports parallel scoring by splitting output.jsonl into chunks:
    python -m nemo_skills.dataset.nv_tts.scripts.score \
        --results_dir /path --benchmark nv_tts.libritts_seen \
        --num_chunks 8 --chunk_id 0

Each chunk automatically attempts to merge all chunks after finishing its own
scoring (self-aggregation). Only the last chunk to complete performs the merge.

Manual merge is also supported:
    python -m nemo_skills.dataset.nv_tts.scripts.score \
        --results_dir /path --benchmark nv_tts.libritts_seen \
        --merge_scoring_chunks --num_chunks 8 \
        --codec_model_path nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps
"""

import argparse
import json
import os
import tempfile



def _import_nemo_evaluate():
    """Lazy import of NeMo evaluate functions (requires NeMo + GPU environment)."""
    from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import (
        compute_global_metrics,
        evaluate,
    )

    return evaluate, compute_global_metrics


def _get_chunk_output_path(benchmark_dir: str, chunk_id: int) -> str:
    """Get the output path for a scoring chunk."""
    return os.path.join(benchmark_dir, f"output_with_metrics_chunk_{chunk_id}.jsonl")


def _read_output_jsonl(output_jsonl: str):
    """Read output.jsonl and return (records, entries) lists."""
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
            codec_codes_path = record.get("debug_info", {}).get("codec_codes_path")
            if audio_path and manifest_entry:
                entries.append((manifest_entry, audio_path, codec_codes_path))
    return records, entries


def run_scoring(
    results_dir: str,
    sv_model: str = "titanet",
    asr_model_name: str = "nvidia/parakeet-tdt-1.1b",
    language: str = "en",
    with_utmosv2: bool = False,
    codec_model_path: str = None,
    benchmark: str = None,
    num_chunks: int = None,
    chunk_id: int = None,
) -> None:
    """Run NeMo scoring on benchmarks in results_dir.

    FCD is always enabled. In chunked mode it is skipped per-chunk and
    computed during the merge step instead.

    When running in chunked mode, each chunk automatically attempts to merge
    all chunks after finishing its own scoring (self-aggregation pattern).
    Only the last chunk to complete will actually perform the merge, since
    merge_scoring_chunks checks for all .done files and metrics.json existence.

    Args:
        benchmark: If provided, score only this benchmark. Otherwise score all.
        num_chunks: If set, split the data into this many chunks.
        chunk_id: If set (along with num_chunks), score only this chunk.
    """
    from nemo_skills.file_utils import calculate_chunk_indices

    evaluate, _ = _import_nemo_evaluate()

    benchmarks_dir = os.path.join(results_dir, "eval-results")
    if not os.path.exists(benchmarks_dir):
        benchmarks_dir = results_dir

    # When running in chunked mode, skip FCD (computed in aggregation)
    is_chunked = num_chunks is not None and chunk_id is not None

    scoring_cfg = {
        "sv_model": sv_model,
        "asr_model_name": asr_model_name,
        "language": language,
        "with_utmosv2": with_utmosv2,
        "with_fcd": not is_chunked,
        "codec_model_path": codec_model_path,
    }

    # Determine which benchmarks to score
    if benchmark:
        benchmarks_to_score = [benchmark]
    else:
        benchmarks_to_score = os.listdir(benchmarks_dir)

    for bench in benchmarks_to_score:
        benchmark_dir = os.path.join(benchmarks_dir, bench)
        if not os.path.isdir(benchmark_dir):
            continue

        output_jsonl = os.path.join(benchmark_dir, "output.jsonl")
        if not os.path.exists(output_jsonl):
            print(f"Skipping {bench}: output.jsonl not found")
            continue

        if is_chunked:
            # Chunked mode: check if this chunk is already done
            chunk_output_path = _get_chunk_output_path(benchmark_dir, chunk_id)
            done_file = f"{chunk_output_path}.done"
            if os.path.exists(done_file):
                print(f"Skipping {bench} chunk {chunk_id}: already done")
            else:
                print(f"\nScoring: {bench} (chunk {chunk_id}/{num_chunks})")
                _score_benchmark_chunk(
                    output_jsonl, scoring_cfg, benchmark_dir, num_chunks, chunk_id, calculate_chunk_indices
                )

            # Auto-attempt merge (mirrors generation's self-aggregation pattern).
            # merge_scoring_chunks checks all .done files and metrics.json existence,
            # so only the last chunk to finish will actually perform the merge.
            merge_scoring_chunks(
                results_dir=results_dir,
                benchmark=bench,
                num_chunks=num_chunks,
                codec_model_path=codec_model_path,
            )
        else:
            # Non-chunked mode: original behavior
            metrics_path = os.path.join(benchmark_dir, "metrics.json")
            if os.path.exists(metrics_path):
                print(f"Skipping {bench}: metrics.json already exists")
                continue

            print(f"\nScoring: {bench}")
            metrics = score_benchmark(output_jsonl, scoring_cfg)

            # Save metrics.json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved: {metrics_path}")
            print(f"  CER: {metrics.get('cer_cumulative', 'N/A'):.4f}")
            print(f"  WER: {metrics.get('wer_cumulative', 'N/A'):.4f}")
            if "utmosv2_avg" in metrics:
                print(f"  UTMOSv2: {metrics.get('utmosv2_avg', 'N/A'):.4f}")


def _score_benchmark_chunk(
    output_jsonl: str,
    scoring_cfg: dict,
    benchmark_dir: str,
    num_chunks: int,
    chunk_id: int,
    calculate_chunk_indices,
) -> None:
    """Score a single chunk of a benchmark and write per-file metrics."""
    records, entries = _read_output_jsonl(output_jsonl)

    if not entries:
        return

    # Compute chunk slice
    start_idx, end_idx = calculate_chunk_indices(len(records), num_chunks, chunk_id)
    chunk_records = records[start_idx:end_idx]
    chunk_entries = entries[start_idx:end_idx]

    if not chunk_entries:
        print(f"  Chunk {chunk_id}: no entries to score")
        return

    print(f"  Chunk {chunk_id}: scoring items {start_idx}..{end_idx} ({len(chunk_entries)} items)")

    # Create temp dir with manifest and symlinks for this chunk
    with tempfile.TemporaryDirectory(prefix=f"tts_scoring_chunk{chunk_id}_") as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        gen_audio_dir = os.path.join(tmp_dir, "generated")
        os.makedirs(gen_audio_dir)

        with open(manifest_path, "w") as f:
            for i, (manifest_entry, audio_path, codec_codes_path) in enumerate(chunk_entries):
                f.write(json.dumps(manifest_entry) + "\n")
                # Symlink audio file
                audio_dst = os.path.join(gen_audio_dir, f"predicted_audio_{i}.wav")
                if os.path.exists(audio_path):
                    os.symlink(audio_path, audio_dst)
                # Symlink codec codes file if available (for FCD scoring)
                if codec_codes_path and os.path.exists(codec_codes_path):
                    codec_codes_dst = os.path.join(gen_audio_dir, f"predicted_codes_{i}.pt")
                    os.symlink(codec_codes_path, codec_codes_dst)

        avg_metrics, filewise_metrics = evaluate(
            manifest_path=manifest_path,
            audio_dir=None,
            generated_audio_dir=gen_audio_dir,
            language=scoring_cfg.get("language", "en"),
            sv_model_type=scoring_cfg.get("sv_model", "titanet"),
            asr_model_name=scoring_cfg.get("asr_model_name", "nvidia/parakeet-tdt-1.1b"),
            with_utmosv2=scoring_cfg.get("with_utmosv2", False),
            with_fcd=False,  # FCD is always skipped in chunked mode
            codec_model_path=None,
        )

        # Write per-file metrics to chunk output file
        chunk_output_path = _get_chunk_output_path(benchmark_dir, chunk_id)
        with open(chunk_output_path, "w") as f:
            for i, record in enumerate(chunk_records):
                if i < len(filewise_metrics):
                    record["metrics"] = filewise_metrics[i]
                f.write(json.dumps(record) + "\n")

        # Touch .done file
        done_file = f"{chunk_output_path}.done"
        open(done_file, "w").close()

        print(f"  Saved: {chunk_output_path}")
        print(f"  Chunk {chunk_id} done.")


def score_benchmark(output_jsonl: str, scoring_cfg: dict) -> dict:
    """Score a single benchmark (non-chunked, original behavior)."""
    records, entries = _read_output_jsonl(output_jsonl)

    if not entries:
        return {}

    # Create temp dir with manifest and symlinks
    with tempfile.TemporaryDirectory(prefix="tts_scoring_") as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        gen_audio_dir = os.path.join(tmp_dir, "generated")
        os.makedirs(gen_audio_dir)

        with open(manifest_path, "w") as f:
            for i, (manifest_entry, audio_path, codec_codes_path) in enumerate(entries):
                f.write(json.dumps(manifest_entry) + "\n")
                # Symlink audio file
                audio_dst = os.path.join(gen_audio_dir, f"predicted_audio_{i}.wav")
                if os.path.exists(audio_path):
                    os.symlink(audio_path, audio_dst)
                # Symlink codec codes file if available (for FCD scoring)
                if codec_codes_path and os.path.exists(codec_codes_path):
                    codec_codes_dst = os.path.join(gen_audio_dir, f"predicted_codes_{i}.pt")
                    os.symlink(codec_codes_path, codec_codes_dst)

        avg_metrics, filewise_metrics = evaluate(
            manifest_path=manifest_path,
            audio_dir=None,
            generated_audio_dir=gen_audio_dir,
            language=scoring_cfg.get("language", "en"),
            sv_model_type=scoring_cfg.get("sv_model", "titanet"),
            asr_model_name=scoring_cfg.get("asr_model_name", "nvidia/parakeet-tdt-1.1b"),
            with_utmosv2=scoring_cfg.get("with_utmosv2", False),
            with_fcd=scoring_cfg.get("with_fcd", False),
            codec_model_path=scoring_cfg.get("codec_model_path"),
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


def merge_scoring_chunks(
    results_dir: str,
    benchmark: str,
    num_chunks: int,
    codec_model_path: str = None,
) -> None:
    """Merge per-chunk scoring outputs and compute global metrics.

    Reads output_with_metrics_chunk_*.jsonl files, concatenates them into
    output_with_metrics.jsonl, recomputes global metrics (cumulative WER/CER,
    averages), and computes FCD. Writes metrics.json.
    """
    _, compute_global_metrics = _import_nemo_evaluate()

    benchmarks_dir = os.path.join(results_dir, "eval-results")
    if not os.path.exists(benchmarks_dir):
        benchmarks_dir = results_dir

    if benchmark:
        benchmarks_to_merge = [benchmark]
    else:
        benchmarks_to_merge = sorted(os.listdir(benchmarks_dir))

    for bench in benchmarks_to_merge:
        benchmark_dir = os.path.join(benchmarks_dir, bench)
        if not os.path.isdir(benchmark_dir):
            continue

        metrics_path = os.path.join(benchmark_dir, "metrics.json")
        if os.path.exists(metrics_path):
            print(f"Skipping {bench}: metrics.json already exists")
            continue

        print(f"\nMerging scoring chunks for: {bench}")

        # Check all chunks are done
        all_done = True
        for chunk_id in range(num_chunks):
            chunk_path = _get_chunk_output_path(benchmark_dir, chunk_id)
            done_file = f"{chunk_path}.done"
            if not os.path.exists(done_file):
                print(f"  WARNING: chunk {chunk_id} not done ({done_file} missing)")
                all_done = False
        if not all_done:
            print(f"  Skipping {bench}: not all chunks are done")
            continue

        # Read and concatenate all chunk outputs
        all_records = []
        all_filewise_metrics = []
        gt_audio_paths = []
        predicted_codes_paths = []

        for chunk_id in range(num_chunks):
            chunk_path = _get_chunk_output_path(benchmark_dir, chunk_id)
            with open(chunk_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    all_records.append(record)
                    if "metrics" in record:
                        all_filewise_metrics.append(record["metrics"])
                        # Collect paths for FCD
                        gt_path = record["metrics"].get("gt_audio_filepath")
                        if gt_path:
                            gt_audio_paths.append(gt_path)
                        # Get codec codes path from debug_info in original record
                        codes_path = record.get("debug_info", {}).get("codec_codes_path")
                        if codes_path:
                            predicted_codes_paths.append(codes_path)

        if not all_filewise_metrics:
            print(f"  No metrics found in chunks for {bench}")
            continue

        print(f"  Merged {len(all_filewise_metrics)} scored items from {num_chunks} chunks")

        # Compute global metrics using the NeMo helper (always includes FCD)
        fcd_gt_paths = gt_audio_paths if len(gt_audio_paths) == len(all_filewise_metrics) else None
        fcd_codes_paths = (
            predicted_codes_paths if len(predicted_codes_paths) == len(all_filewise_metrics) else None
        )

        avg_metrics = compute_global_metrics(
            filewise_metrics=all_filewise_metrics,
            gt_audio_paths=fcd_gt_paths,
            predicted_codes_paths=fcd_codes_paths,
            codec_model_path=codec_model_path,
        )

        # Write merged output_with_metrics.jsonl
        output_with_metrics_path = os.path.join(benchmark_dir, "output_with_metrics.jsonl")
        with open(output_with_metrics_path, "w") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")
        print(f"  Saved: {output_with_metrics_path}")

        # Write metrics.json
        with open(metrics_path, "w") as f:
            json.dump(avg_metrics, f, indent=2)
        print(f"  Saved: {metrics_path}")
        print(f"    CER: {avg_metrics.get('cer_cumulative', 'N/A'):.4f}")
        print(f"    WER: {avg_metrics.get('wer_cumulative', 'N/A'):.4f}")
        if "utmosv2_avg" in avg_metrics:
            print(f"    UTMOSv2: {avg_metrics.get('utmosv2_avg', 'N/A'):.4f}")
        if "frechet_codec_distance" in avg_metrics:
            print(f"    FCD: {avg_metrics.get('frechet_codec_distance', 'N/A')}")

        # Clean up chunk files
        for chunk_id in range(num_chunks):
            chunk_path = _get_chunk_output_path(benchmark_dir, chunk_id)
            done_file = f"{chunk_path}.done"
            for f_path in [chunk_path, done_file]:
                if os.path.exists(f_path):
                    os.remove(f_path)
        print(f"  Cleaned up chunk files")


def _format_value(value, format_spec: str = ".4f", na_str: str = "N/A") -> str:
    """Format a value for display."""
    if value is None:
        return na_str
    try:
        return f"{value:{format_spec}}"
    except (ValueError, TypeError):
        return str(value)


def _format_percent(value, na_str: str = "N/A") -> str:
    """Format a value as percentage."""
    if value is None:
        return na_str
    try:
        return f"{value * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


# TTS metrics: (key, display_name, format_func, higher_is_better)
_TTS_METRICS = [
    ("wer_cumulative", "WER (cumulative)", _format_percent, False),
    ("cer_cumulative", "CER (cumulative)", _format_percent, False),
    ("wer_filewise_avg", "WER (filewise avg)", _format_percent, False),
    ("cer_filewise_avg", "CER (filewise avg)", _format_percent, False),
    ("utmosv2_avg", "UTMOS v2", lambda v: _format_value(v, ".3f"), True),
    ("ssim_pred_gt_avg", "SSIM (pred vs GT)", lambda v: _format_value(v, ".4f"), True),
    ("ssim_pred_context_avg", "SSIM (pred vs context)", lambda v: _format_value(v, ".4f"), True),
    ("total_gen_audio_seconds", "Total audio (sec)", lambda v: _format_value(v, ".1f"), None),
    ("frechet_codec_distance", "FCD", lambda v: _format_value(v, ".4f"), False),
]


def _generate_markdown_report(
    results_dir: str,
    benchmark_metrics: dict[str, dict],
) -> str:
    """Generate a Markdown summary report for a single model's eval results.

    Args:
        results_dir: Path to the results directory (used in report title).
        benchmark_metrics: Mapping of benchmark name -> metrics dict.

    Returns:
        The Markdown report as a string.
    """
    model_name = os.path.basename(results_dir.rstrip("/"))
    lines = []
    lines.append(f"# TTS Evaluation Report: {model_name}\n")

    # Summary section: average across all benchmarks
    if len(benchmark_metrics) > 1:
        lines.append("## Summary (Averaged Across Test Sets)\n")
        # Compute averages
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for m in benchmark_metrics.values():
            for key, _, _, _ in _TTS_METRICS:
                if key in m and m[key] is not None:
                    totals[key] = totals.get(key, 0) + m[key]
                    counts[key] = counts.get(key, 0) + 1
        averages = {k: totals[k] / counts[k] for k in totals if counts.get(k, 0) > 0}

        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for metric_key, display_name, fmt_func, _ in _TTS_METRICS:
            if metric_key == "total_gen_audio_seconds":
                continue
            val = averages.get(metric_key)
            if val is not None:
                lines.append(f"| {display_name} | {fmt_func(val)} |")
        lines.append("")

    # Per-benchmark sections
    lines.append("## Per-Test-Set Results\n")
    for benchmark in sorted(benchmark_metrics):
        m = benchmark_metrics[benchmark]
        lines.append(f"### {benchmark}\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for metric_key, display_name, fmt_func, _ in _TTS_METRICS:
            val = m.get(metric_key)
            if val is not None:
                lines.append(f"| {display_name} | {fmt_func(val)} |")
        lines.append("")

    lines.append("---")
    lines.append("*Lower WER/CER/FCD is better, higher UTMOS/SSIM is better.*")

    return "\n".join(lines)


def run_aggregation(results_dir: str) -> None:
    """Print summary of all metrics and generate a Markdown report."""
    benchmarks_dir = os.path.join(results_dir, "eval-results")
    if not os.path.exists(benchmarks_dir):
        benchmarks_dir = results_dir

    benchmark_metrics: dict[str, dict] = {}

    print("\nAggregated Results:")
    for benchmark in sorted(os.listdir(benchmarks_dir)):
        metrics_path = os.path.join(benchmarks_dir, benchmark, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            benchmark_metrics[benchmark] = metrics
            print(f"  {benchmark}:")
            print(f"    CER: {metrics.get('cer_cumulative', 'N/A'):.4f}")
            print(f"    WER: {metrics.get('wer_cumulative', 'N/A'):.4f}")
            if "utmosv2_avg" in metrics:
                print(f"    UTMOSv2: {metrics.get('utmosv2_avg', 'N/A'):.4f}")

    if benchmark_metrics:
        report = _generate_markdown_report(results_dir, benchmark_metrics)
        report_path = os.path.join(results_dir, "tts_eval_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nMarkdown report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Scoring")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--sv_model", default="titanet")
    parser.add_argument("--asr_model_name", default="nvidia/parakeet-tdt-1.1b")
    parser.add_argument("--language", default="en")
    parser.add_argument("--with_utmosv2", action="store_true")
    parser.add_argument("--codec_model_path", default=None, help="Path to codec model for FCD scoring")
    parser.add_argument("--aggregation_only", action="store_true")
    parser.add_argument("--benchmark", default=None, help="Score only this benchmark (e.g. nv_tts.libritts_seen)")
    # Chunked scoring arguments
    parser.add_argument("--num_chunks", type=int, default=None, help="Number of chunks to split scoring into")
    parser.add_argument("--chunk_id", type=int, default=None, help="Chunk ID to score (0-indexed)")
    # Merge chunks argument
    parser.add_argument(
        "--merge_scoring_chunks",
        action="store_true",
        help="Merge chunked scoring outputs and compute global metrics",
    )
    args = parser.parse_args()

    if args.aggregation_only:
        run_aggregation(args.results_dir)
    elif args.merge_scoring_chunks:
        if args.num_chunks is None:
            parser.error("--num_chunks is required when using --merge_scoring_chunks")
        merge_scoring_chunks(
            results_dir=args.results_dir,
            benchmark=args.benchmark,
            num_chunks=args.num_chunks,
            codec_model_path=args.codec_model_path,
        )
    else:
        run_scoring(
            args.results_dir,
            sv_model=args.sv_model,
            asr_model_name=args.asr_model_name,
            language=args.language,
            with_utmosv2=args.with_utmosv2,
            codec_model_path=args.codec_model_path,
            benchmark=args.benchmark,
            num_chunks=args.num_chunks,
            chunk_id=args.chunk_id,
        )
