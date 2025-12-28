#!/usr/bin/env python3
"""
Compare multiple TTS evaluation results and generate a Markdown report.

Usage:
    python compare_tts_eval_results.py \
        --eval_folders "/path/to/eval1:Model A" "/path/to/eval2:Model B" \
        --output report.md

Supports remote folders via SSH:
    python compare_tts_eval_results.py \
        --eval_folders "host1:/path/to/eval1:Model A" "host2:/path/to/eval2:Model B" \
        --output report.md

Each eval folder should contain subdirectories for different test sets,
each with a metrics.json file.
"""

import argparse
import json
import os
import subprocess
from typing import Optional


def run_remote_cmd(host: str, cmd: str, timeout: int = 30) -> Optional[str]:
    """Run a command on a remote host via SSH."""
    try:
        result = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return None


def list_test_sets_local(eval_folder: str) -> list[str]:
    """List test set subdirectories in a local eval folder."""
    if not os.path.isdir(eval_folder):
        return []
    return [d for d in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, d))]


def list_test_sets_remote(host: str, eval_folder: str) -> list[str]:
    """List test set subdirectories in a remote eval folder via SSH."""
    output = run_remote_cmd(host, f"ls -1 {eval_folder}")
    if output is None:
        return []
    return [d.strip() for d in output.split("\n") if d.strip()]


def load_metrics_local(metrics_path: str) -> Optional[dict]:
    """Load metrics from a local JSON file."""
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_metrics_remote(host: str, metrics_path: str) -> Optional[dict]:
    """Load metrics from a remote JSON file via SSH."""
    output = run_remote_cmd(host, f"cat {metrics_path}")
    if output is None:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def load_metrics(path: str, host: Optional[str] = None) -> Optional[dict]:
    """Load metrics from a local or remote JSON file."""
    if host:
        return load_metrics_remote(host, path)
    return load_metrics_local(path)


def list_test_sets(eval_folder: str, host: Optional[str] = None) -> list[str]:
    """List test set subdirectories."""
    if host:
        return list_test_sets_remote(host, eval_folder)
    return list_test_sets_local(eval_folder)


def format_value(value, format_spec: str = ".4f", na_str: str = "N/A") -> str:
    """Format a value for display."""
    if value is None:
        return na_str
    try:
        return f"{value:{format_spec}}"
    except (ValueError, TypeError):
        return str(value)


def format_percent(value, na_str: str = "N/A") -> str:
    """Format a value as percentage."""
    if value is None:
        return na_str
    try:
        return f"{value * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def parse_folder_spec(spec: str) -> tuple[Optional[str], str, str]:
    """
    Parse folder specification in format:
      - 'path:name' (local)
      - 'path' (local, auto-name)
      - 'host:path:name' (remote)
      - 'host:path' (remote, auto-name)
    """
    parts = spec.split(":")
    if len(parts) == 1:
        path = parts[0]
        name = os.path.basename(path.rstrip("/"))
        return None, path, name
    if len(parts) == 2:
        if parts[0].startswith("/") or parts[0].startswith("."):
            return None, parts[0], parts[1]
        else:
            host, path = parts[0], parts[1]
            name = os.path.basename(path.rstrip("/"))
            return host, path, name
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) > 3:
        host = parts[0]
        name = parts[-1]
        path = ":".join(parts[1:-1])
        return host, path, name
    return None, spec, os.path.basename(spec.rstrip("/"))


# TTS metrics: (key, display_name, format_func, higher_is_better)
TTS_METRICS = [
    ("wer_cumulative", "WER (cumulative)", format_percent, False),
    ("cer_cumulative", "CER (cumulative)", format_percent, False),
    ("wer_filewise_avg", "WER (filewise avg)", format_percent, False),
    ("cer_filewise_avg", "CER (filewise avg)", format_percent, False),
    ("utmosv2_avg", "UTMOS v2", lambda v: format_value(v, ".3f"), True),
    ("ssim_pred_gt_avg", "SSIM (pred vs GT)", lambda v: format_value(v, ".4f"), True),
    ("ssim_pred_context_avg", "SSIM (pred vs context)", lambda v: format_value(v, ".4f"), True),
    ("total_gen_audio_seconds", "Total audio (sec)", lambda v: format_value(v, ".1f"), None),
]


def generate_test_set_table(
    test_set: str,
    models: list[tuple[str, dict]],
) -> list[str]:
    """Generate a comparison table for a single test set."""
    lines = []
    lines.append(f"### {test_set}\n")

    # Table header
    header = "| Metric | " + " | ".join(name for name, _ in models) + " |"
    separator = "|" + "|".join(["---"] * (len(models) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    for metric_key, display_name, fmt_func, higher_better in TTS_METRICS:
        values = []
        raw_values = []
        for _, m in models:
            val = m.get(metric_key)
            raw_values.append(val)
            values.append(fmt_func(val))

        # Highlight best value
        if higher_better is not None:
            valid_vals = [(i, v) for i, v in enumerate(raw_values) if v is not None]
            if len(valid_vals) >= 2:
                if higher_better:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                values[best_idx] = f"**{values[best_idx]}**"

        row = f"| {display_name} | " + " | ".join(values) + " |"
        lines.append(row)

    lines.append("")
    return lines


def compute_summary_metrics(
    all_test_sets: list[str],
    model_metrics: dict[str, dict[str, dict]],
) -> dict[str, dict[str, float]]:
    """Compute average metrics across all test sets for each model."""
    summary = {}
    for model_name, test_data in model_metrics.items():
        totals = {}
        counts = {}
        for test_set in all_test_sets:
            if test_set not in test_data:
                continue
            m = test_data[test_set]
            for key, _, _, _ in TTS_METRICS:
                if key in m and m[key] is not None:
                    totals[key] = totals.get(key, 0) + m[key]
                    counts[key] = counts.get(key, 0) + 1
        summary[model_name] = {k: totals[k] / counts[k] for k in totals if counts.get(k, 0) > 0}
    return summary


def determine_best_model(models: list[tuple[str, dict]]) -> tuple[str, str]:
    """Determine the best model based on summary metrics."""
    if len(models) < 2:
        return "", "Need at least 2 models to compare."

    scores = {name: 0.0 for name, _ in models}
    comparisons = []

    # Weight metrics
    weights = {
        "wer_cumulative": 2.0,
        "cer_cumulative": 1.5,
        "utmosv2_avg": 2.0,
        "ssim_pred_gt_avg": 1.0,
    }

    for metric_key, metric_name, _, higher_better in TTS_METRICS:
        if higher_better is None:
            continue
        weight = weights.get(metric_key, 1.0)
        valid = [(name, m.get(metric_key)) for name, m in models if m.get(metric_key) is not None]
        if len(valid) < 2:
            continue

        if higher_better:
            best_val = max(v for _, v in valid)
            worst_val = min(v for _, v in valid)
        else:
            best_val = min(v for _, v in valid)
            worst_val = max(v for _, v in valid)

        if best_val == worst_val:
            continue

        best_name = [n for n, v in valid if v == best_val][0]
        for name, val in valid:
            if higher_better:
                normalized = (val - worst_val) / (best_val - worst_val)
            else:
                normalized = (worst_val - val) / (worst_val - best_val)
            scores[name] += normalized * weight
        comparisons.append(f"{best_name} leads in {metric_name}")

    if not any(scores.values()):
        return "", "Unable to determine best model: insufficient metrics."

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name = sorted_scores[0][0]
    return best_name, (
        f"**{best_name}** performs best overall. "
        f"Key advantages: {'; '.join(comparisons[:4]) if comparisons else 'balanced performance'}."
    )


def generate_report(
    model_names: list[str],
    all_test_sets: list[str],
    model_metrics: dict[str, dict[str, dict]],
    output_path: str,
):
    """Generate the full Markdown comparison report."""
    lines = []
    lines.append("# TTS Evaluation Comparison Report\n")
    lines.append(f"Comparing {len(model_names)} model(s): " + ", ".join(model_names) + "\n")

    # Summary section
    lines.append("## Summary (Averaged Across Test Sets)\n")
    summary = compute_summary_metrics(all_test_sets, model_metrics)
    summary_models = [(name, summary.get(name, {})) for name in model_names]

    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|" + "|".join(["---"] * (len(model_names) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    for metric_key, display_name, fmt_func, higher_better in TTS_METRICS:
        if metric_key == "total_gen_audio_seconds":
            continue  # Skip total audio in summary
        values = []
        raw_values = []
        for name in model_names:
            val = summary.get(name, {}).get(metric_key)
            raw_values.append(val)
            values.append(fmt_func(val))

        if higher_better is not None:
            valid_vals = [(i, v) for i, v in enumerate(raw_values) if v is not None]
            if len(valid_vals) >= 2:
                if higher_better:
                    best_idx = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_vals, key=lambda x: x[1])[0]
                values[best_idx] = f"**{values[best_idx]}**"

        row = f"| {display_name} | " + " | ".join(values) + " |"
        lines.append(row)

    lines.append("")

    # Best model analysis
    best_name, explanation = determine_best_model(summary_models)
    lines.append("### Analysis\n")
    lines.append(explanation)
    lines.append("")

    # Per-test-set sections
    lines.append("## Per-Test-Set Results\n")

    for test_set in sorted(all_test_sets):
        test_models = []
        for name in model_names:
            m = model_metrics.get(name, {}).get(test_set)
            if m is not None:
                test_models.append((name, m))

        if not test_models:
            continue

        lines.extend(generate_test_set_table(test_set, test_models))

    # Legend
    lines.append("---")
    lines.append("*Lower WER/CER is better, higher UTMOS/SSIM is better. **bold** = best value.*")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")
    print("\n" + "=" * 60)
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Compare TTS evaluation results from multiple folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Remote folders (via SSH):
    python compare_tts_eval_results.py \\
        --eval_folders "login-eos:/path/to/eval1/eval-results:Model A" \\
                       "login-eos:/path/to/eval2/eval-results:Model B" \\
        --output comparison_report.md

    # Local folders:
    python compare_tts_eval_results.py \\
        --eval_folders "/path/to/eval1:Model A" "/path/to/eval2:Model B" \\
        --output comparison_report.md

Each eval folder should contain subdirectories for different test sets
(e.g., nv_tts.vctk, nv_tts.libritts_test_clean), each with a metrics.json.
        """,
    )
    parser.add_argument(
        "--eval_folders",
        nargs="+",
        required=True,
        help="Evaluation folders: 'path:name', 'host:path:name', or 'host:path'",
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "tts_comparison_report.md")
    parser.add_argument("--output", type=str, default=default_output, help="Output Markdown file path")

    args = parser.parse_args()

    # Parse folder specs
    folder_specs = []
    for spec in args.eval_folders:
        host, path, name = parse_folder_spec(spec)
        folder_specs.append((host, path, name))

    # Discover all test sets across all models
    all_test_sets = set()
    model_test_sets = {}
    for host, path, name in folder_specs:
        test_sets = list_test_sets(path, host)
        model_test_sets[name] = (host, path, test_sets)
        all_test_sets.update(test_sets)
        loc = f"{host}:{path}" if host else path
        print(f"Found {len(test_sets)} test sets for {name} at {loc}")

    if not all_test_sets:
        print("Error: No test sets found.")
        return 1

    # Load metrics for each model and test set
    model_metrics: dict[str, dict[str, dict]] = {}
    model_names = []

    for host, path, name in folder_specs:
        model_names.append(name)
        model_metrics[name] = {}
        test_sets = model_test_sets[name][2]

        for test_set in test_sets:
            metrics_path = f"{path}/{test_set}/metrics.json"
            metrics = load_metrics(metrics_path, host)
            if metrics is not None:
                model_metrics[name][test_set] = metrics
                print(f"  Loaded {test_set} for {name}")
            else:
                print(f"  Skipping {test_set} for {name} (no metrics.json)")

    # Filter to test sets that have metrics for at least one model
    valid_test_sets = [ts for ts in all_test_sets if any(ts in model_metrics.get(n, {}) for n in model_names)]

    if not valid_test_sets:
        print("Error: No valid metrics found for any test set.")
        return 1

    generate_report(model_names, valid_test_sets, model_metrics, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
