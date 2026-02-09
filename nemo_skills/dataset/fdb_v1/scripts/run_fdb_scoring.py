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
FDB scoring: prepare (copy audio to fdb_prepared) -> run ASR -> run FDB evaluate -> write metrics.json.
Used by run_eval.py.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ASR_TASK_MAP = {
    "pause": "full",
    "backchannel": "full",
    "turn_taking": "full",
    "interruption": "user_interruption",
    "background_speech": "full",
    "talking_to_other": "full",
}
FDB_TASK_MAP = {
    "pause": "pause_handling",
    "backchannel": "backchannel",
    "turn_taking": "smooth_turn_taking",
    "interruption": "user_interruption",
    "background_speech": "background_speech",
    "talking_to_other": "talking_to_other",
}


def main():
    parser = argparse.ArgumentParser(description="FDB prepare + ASR + evaluate -> metrics.json")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument("--fdb_repo", type=Path, required=True)
    parser.add_argument("--subtest", required=True, choices=list(ASR_TASK_MAP))
    parser.add_argument("--fdb_data_path", type=Path, default=None, help="FDB dataset root; required for turn_taking (turn_taking.json) and interruption (interrupt.json)")
    parser.add_argument("--fdb_version", default="v1.0", choices=["v1.0", "v1.5"], help="FDB dataset version (metadata paths and metrics key)")
    args = parser.parse_args()

    eval_results_dir = args.eval_results_dir.resolve()
    fdb_repo = args.fdb_repo.resolve()
    metrics_file = eval_results_dir / "metrics.json"
    if not eval_results_dir.exists() or not fdb_repo.exists():
        print("Error: eval_results_dir or fdb_repo not found.")
        sys.exit(1)
    if not (eval_results_dir / "output.jsonl").exists():
        print("Error: output.jsonl not found.")
        sys.exit(1)

    # turn_taking and interruption need fdb_data_path so prepare can copy turn_taking.json / interrupt.json
    if args.subtest in ("turn_taking", "interruption") and (args.fdb_data_path is None or not args.fdb_data_path.exists()):
        print(
            f"Error: --fdb_data_path is required for subtest '{args.subtest}' "
            "(FDB dataset root; used to copy turn_taking.json or interrupt.json into each sample dir)."
        )
        sys.exit(1)

    asr_task = ASR_TASK_MAP[args.subtest]
    fdb_task = FDB_TASK_MAP[args.subtest]
    prep_script = Path(__file__).resolve().parent / "prepare_fdb_eval_dir.py"

    prep_cmd = [
        sys.executable, str(prep_script),
        "--eval_results_dir", str(eval_results_dir),
        "--fdb_repo", str(fdb_repo),
        "--run_asr", "--asr_task", asr_task,
        "--subtest", args.subtest,
        "--fdb_version", args.fdb_version,
    ]
    if args.fdb_data_path is not None:
        prep_cmd.extend(["--fdb_data_path", str(args.fdb_data_path)])
    subprocess.run(prep_cmd, check=True)

    fdb_prepared = eval_results_dir / "fdb_prepared"
    evaluate_script = fdb_repo / "evaluation" / "evaluate.py"
    if not evaluate_script.exists():
        print(f"Error: {evaluate_script} not found")
        sys.exit(1)
    # Run from evaluation/ so FDB scripts find ./icc_gt_distribution.json (backchannel) and other relative paths.
    # Pass through env so NVIDIA_API_KEY is available for interruption/behavior tasks (NVIDIA NIM API).
    result = subprocess.run(
        [sys.executable, str(evaluate_script), "--task", fdb_task, "--root_dir", str(fdb_prepared)],
        cwd=str(fdb_repo / "evaluation"), capture_output=True, text=True, env=os.environ.copy(),
    )
    stdout, stderr = result.stdout, result.stderr
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    metrics = {}
    combined = stdout + "\n" + stderr
    # Only extract explicitly known FDB metric lines (no generic regex)
    explicit_metrics = [
        ("JSD - Mean", "jsd"),
        ("TOR - Mean", "tor"),
        ("Frequency - Mean", "frequency"),
        ("Average take turn", "turn"),
        ("Average latency", "latency"),
    ]
    for name, key in explicit_metrics:
        m = re.search(rf"{re.escape(name)}\s*:\s*([0-9.]+)", combined)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass
    if not metrics:
        metrics["status"] = "no_metrics_found"

    benchmark_key = f"fdb_v1_5.{args.subtest}" if args.fdb_version == "v1.5" else f"fdb_v1.{args.subtest}"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump({benchmark_key: {"greedy": metrics}}, f, indent=2)
    print(f"Metrics written to {metrics_file}")


if __name__ == "__main__":
    main()