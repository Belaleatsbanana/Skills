# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Run Full-Duplex-Bench scoring with nemo-skills compatible output structure.

Creates:
- summarized-results/ directory
- metrics.json with evaluation results
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_scoring(
    eval_results_dir: str,
    fdb_repo: str,
    subtest: str,
    force: bool = False,
):
    """Run Full-Duplex-Bench scoring and save results in nemo-skills format."""
    eval_results_dir = Path(eval_results_dir)
    output_jsonl = eval_results_dir / "output.jsonl"
    converted_jsonl = eval_results_dir / "fdb_format.jsonl"
    summarized_dir = eval_results_dir / "summarized-results"
    metrics_file = eval_results_dir / "metrics.json"

    # Skip if already scored (unless force is set)
    if metrics_file.exists() and not force:
        print(f"Scoring already done for fullduplexbench.{subtest} (metrics.json exists). Skipping.")
        print("Use --force to re-run scoring.")
        with open(metrics_file) as f:
            existing_metrics = json.load(f)
        print(f"Existing metrics: {json.dumps(existing_metrics, indent=2)}")
        return 0

    # Create summarized-results directory
    summarized_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert format
    print(f"Converting {output_jsonl} to Full-Duplex-Bench format...")
    convert_script = Path(__file__).parent / "convert_to_fdb_format.py"
    cmd = f"python {convert_script} --input {output_jsonl} --output {converted_jsonl} --subtest {subtest}"
    subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run Full-Duplex-Bench evaluation
    print(f"Running Full-Duplex-Bench evaluation for {subtest}...")
    
    # Map our subtest names to FDB task names
    task_mapping = {
        "pause": "pause_handling",
        "backchannel": "backchannel",
        "turn_taking": "smooth_turn_taking",
        "interruption": "user_interruption",
    }
    fdb_task = task_mapping.get(subtest, subtest)
    
    # Full-Duplex-Bench uses evaluation/evaluate.py with --task and --root_dir
    evaluate_script = f"{fdb_repo}/evaluation/evaluate.py"
    cmd = f"cd {fdb_repo} && python {evaluate_script} --task {fdb_task} --root_dir {eval_results_dir}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse metrics from output
    # FDB typically outputs metrics in JSON format or as key-value pairs
    metrics = {}
    
    # Try to parse JSON output
    try:
        # Look for JSON in stdout
        json_match = re.search(r'\{[^}]+\}', result.stdout)
        if json_match:
            metrics = json.loads(json_match.group())
    except Exception:
        pass

    # If no JSON found, try to parse key-value pairs
    if not metrics:
        # Look for patterns like "metric_name: value"
        for line in result.stdout.split("\n"):
            match = re.match(r'(\w+):\s*([\d.]+)', line)
            if match:
                metric_name, metric_value = match.groups()
                try:
                    metrics[metric_name] = float(metric_value)
                except ValueError:
                    metrics[metric_name] = metric_value

    # If still no metrics, check for specific FDB metrics
    if not metrics:
        # Common FDB metrics: takeover_rate, jsd (Jensen-Shannon Divergence), etc.
        for metric in ["takeover_rate", "jsd", "accuracy", "f1_score"]:
            match = re.search(rf'{metric}[:\s]+([0-9.]+)', result.stdout, re.IGNORECASE)
            if match:
                metrics[metric] = float(match.group(1))

    # Save metrics.json in nemo-skills format
    nemo_metrics = {f"fullduplexbench.{subtest}": {"greedy": metrics}}
    with open(metrics_file, "w") as f:
        json.dump(nemo_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    # Also print metrics summary
    print("\n" + "=" * 60)
    print(f"RESULTS for fullduplexbench.{subtest}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Full-Duplex-Bench scoring with nemo-skills output format")
    parser.add_argument(
        "--eval_results_dir", required=True, help="Path to eval-results/fullduplexbench.{subtest}/ directory"
    )
    parser.add_argument("--fdb_repo", required=True, help="Path to Full-Duplex-Bench repository")
    parser.add_argument("--subtest", required=True, help="Subtest name (pause, backchannel, turn_taking, interruption)")
    parser.add_argument("--force", action="store_true", help="Force re-run scoring even if metrics.json exists")

    args = parser.parse_args()

    rc = run_scoring(
        eval_results_dir=args.eval_results_dir,
        fdb_repo=args.fdb_repo,
        subtest=args.subtest,
        force=args.force,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
