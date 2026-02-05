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
Standalone Full-Duplex-Bench scoring script (alternative to run_fdb_scoring.py).

This version provides more detailed output and error handling for debugging.
Useful when running scoring manually or debugging evaluation issues.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def convert_to_fdb_format(input_jsonl: Path, output_jsonl: Path, subtest: str) -> int:
    """Convert nemo-skills format to Full-Duplex-Bench format."""
    convert_script = Path(__file__).parent / "convert_to_fdb_format.py"
    
    cmd = [
        sys.executable,
        str(convert_script),
        "--input", str(input_jsonl),
        "--output", str(output_jsonl),
        "--subtest", subtest,
    ]
    
    print(f"Converting format: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Count entries
    count = 0
    with open(output_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    
    return count


def run_fdb_evaluation(fdb_repo: Path, converted_jsonl: Path, subtest: str) -> dict:
    """Run Full-Duplex-Bench official evaluation."""
    print(f"\n{'='*60}")
    print(f"Running Full-Duplex-Bench evaluation for {subtest}")
    print(f"{'='*60}")
    
    # Full-Duplex-Bench uses evaluation/evaluate.py
    evaluate_script = fdb_repo / "evaluation" / "evaluate.py"
    
    if not evaluate_script.exists():
        raise FileNotFoundError(
            f"Could not find evaluation script at {evaluate_script}. "
            f"Make sure you cloned the Full-Duplex-Bench repo correctly."
        )
    
    # Map our subtest names to FDB task names
    task_mapping = {
        "pause": "pause_handling",
        "backchannel": "backchannel",
        "turn_taking": "smooth_turn_taking",
        "interruption": "user_interruption",
    }
    
    fdb_task = task_mapping.get(subtest, subtest)
    
    # FDB expects --root_dir pointing to a directory with the data
    # We need to prepare the data in the expected format
    eval_results_dir = converted_jsonl.parent
    
    cmd = [
        sys.executable,
        str(evaluate_script),
        "--task", fdb_task,
        "--root_dir", str(eval_results_dir),
    ]
    
    print(f"Evaluation command: {' '.join(cmd)}")
    print(f"Working directory: {fdb_repo}")
    print(f"Note: FDB evaluation expects ASR-aligned transcripts.")
    print(f"      You may need to run prepare_for_eval/asr.py first.")
    
    result = subprocess.run(
        cmd,
        cwd=str(fdb_repo),
        capture_output=True,
        text=True,
    )
    
    print("\n--- STDOUT ---")
    print(result.stdout)
    
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"\n⚠️  Warning: Evaluation returned non-zero exit code: {result.returncode}")
    
    # Parse metrics from output
    metrics = parse_metrics(result.stdout, result.stderr, subtest)
    
    return metrics


def parse_metrics(stdout: str, stderr: str, subtest: str) -> dict:
    """Parse evaluation metrics from FDB output.
    
    Full-Duplex-Bench typically outputs metrics like:
    - takeover_rate: X.XX
    - jsd (Jensen-Shannon Divergence): X.XX
    - accuracy: X.XX
    - f1_score: X.XX
    """
    metrics = {}
    combined_output = stdout + "\n" + stderr
    
    # Try JSON format first
    try:
        json_match = re.search(r'\{[^}]*["\']?\w+["\']?\s*:\s*[\d.]+[^}]*\}', combined_output)
        if json_match:
            potential_json = json_match.group()
            try:
                parsed = json.loads(potential_json)
                if isinstance(parsed, dict):
                    metrics.update(parsed)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"Note: Could not parse as JSON: {e}")
    
    # Parse common metric patterns
    metric_patterns = {
        'takeover_rate': r'takeover[_\s]rate[:\s]+([0-9.]+)',
        'jsd': r'jsd[:\s]+([0-9.]+)',
        'jensen_shannon_divergence': r'jensen[_\s-]shannon[_\s-]divergence[:\s]+([0-9.]+)',
        'accuracy': r'accuracy[:\s]+([0-9.]+)',
        'f1': r'f1[_\s-]?score[:\s]+([0-9.]+)',
        'precision': r'precision[:\s]+([0-9.]+)',
        'recall': r'recall[:\s]+([0-9.]+)',
        'score': r'(?:^|\s)score[:\s]+([0-9.]+)',
    }
    
    for metric_name, pattern in metric_patterns.items():
        if metric_name not in metrics:
            match = re.search(pattern, combined_output, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    pass
    
    # If still no metrics found, try to find any number patterns with labels
    if not metrics:
        print("\n⚠️  Warning: No standard metrics found. Attempting generic extraction...")
        generic_matches = re.findall(r'(\w+)[:\s]+([0-9.]+)', combined_output)
        for name, value in generic_matches:
            if name.lower() not in ['line', 'file', 'error', 'warning']:
                try:
                    metrics[name.lower()] = float(value)
                except ValueError:
                    pass
    
    if not metrics:
        print("\n⚠️  Warning: Could not extract any metrics from FDB output!")
        print("You may need to manually check the output above.")
        metrics['status'] = 'no_metrics_found'
    
    return metrics


def save_metrics(metrics: dict, output_file: Path, subtest: str):
    """Save metrics in nemo-skills format."""
    nemo_metrics = {
        f"fullduplexbench.{subtest}": {
            "greedy": metrics
        }
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(nemo_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_file}")


def print_metrics_summary(metrics: dict, subtest: str):
    """Print formatted metrics summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS for fullduplexbench.{subtest}")
    print(f"{'='*60}")
    
    if not metrics:
        print("  No metrics available")
    else:
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.4f}")
            else:
                print(f"  {key:30s}: {value}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Full-Duplex-Bench scoring with detailed output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a specific subtest
  python run_fdb_scoring_standalone.py \\
    --eval_results_dir /path/to/eval-results/fullduplexbench.pause \\
    --fdb_repo /path/to/Full-Duplex-Bench \\
    --subtest pause
  
  # Force re-scoring
  python run_fdb_scoring_standalone.py \\
    --eval_results_dir /path/to/eval-results/fullduplexbench.turn_taking \\
    --fdb_repo /path/to/Full-Duplex-Bench \\
    --subtest turn_taking \\
    --force
        """
    )
    parser.add_argument(
        "--eval_results_dir",
        type=Path,
        required=True,
        help="Path to eval-results/fullduplexbench.{subtest}/ directory"
    )
    parser.add_argument(
        "--fdb_repo",
        type=Path,
        required=True,
        help="Path to Full-Duplex-Bench repository"
    )
    parser.add_argument(
        "--subtest",
        required=True,
        choices=["pause", "backchannel", "turn_taking", "interruption"],
        help="Subtest name"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run scoring even if metrics.json exists"
    )
    parser.add_argument(
        "--keep_converted",
        action="store_true",
        help="Keep the converted FDB format file after scoring"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.eval_results_dir.exists():
        print(f"Error: Eval results directory not found: {args.eval_results_dir}")
        sys.exit(1)
    
    if not args.fdb_repo.exists():
        print(f"Error: FDB repository not found: {args.fdb_repo}")
        sys.exit(1)
    
    output_jsonl = args.eval_results_dir / "output.jsonl"
    if not output_jsonl.exists():
        print(f"Error: Output file not found: {output_jsonl}")
        print("Make sure generation has completed successfully.")
        sys.exit(1)
    
    converted_jsonl = args.eval_results_dir / "fdb_format.jsonl"
    metrics_file = args.eval_results_dir / "metrics.json"
    
    # Check if already scored
    if metrics_file.exists() and not args.force:
        print(f"✓ Scoring already done (metrics.json exists)")
        print(f"  Use --force to re-run scoring\n")
        with open(metrics_file) as f:
            existing_metrics = json.load(f)
        subtest_key = f"fullduplexbench.{args.subtest}"
        if subtest_key in existing_metrics and "greedy" in existing_metrics[subtest_key]:
            print_metrics_summary(existing_metrics[subtest_key]["greedy"], args.subtest)
        sys.exit(0)
    
    try:
        # Step 1: Convert format
        print(f"\n{'='*60}")
        print("Step 1: Converting to Full-Duplex-Bench format")
        print(f"{'='*60}")
        entry_count = convert_to_fdb_format(output_jsonl, converted_jsonl, args.subtest)
        print(f"✓ Converted {entry_count} entries")
        
        # Step 2: Run FDB evaluation
        print(f"\n{'='*60}")
        print("Step 2: Running Full-Duplex-Bench evaluation")
        print(f"{'='*60}")
        metrics = run_fdb_evaluation(args.fdb_repo, converted_jsonl, args.subtest)
        
        # Step 3: Save metrics
        print(f"\n{'='*60}")
        print("Step 3: Saving metrics")
        print(f"{'='*60}")
        save_metrics(metrics, metrics_file, args.subtest)
        
        # Step 4: Print summary
        print_metrics_summary(metrics, args.subtest)
        
        # Cleanup converted file if requested
        if not args.keep_converted and converted_jsonl.exists():
            converted_jsonl.unlink()
            print(f"✓ Cleaned up converted file: {converted_jsonl}")
        
        print("✓ Scoring completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during scoring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
