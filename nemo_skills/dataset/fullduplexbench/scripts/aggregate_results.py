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
Aggregate Full-Duplex-Bench results across multiple subtests.

This script:
1. Collects metrics from all subtest directories
2. Computes aggregate statistics
3. Generates a summary report
4. Optionally exports to CSV for analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


SUBTESTS = ["pause", "backchannel", "turn_taking", "interruption"]


def load_metrics(eval_results_dir: Path, subtest: str) -> Dict:
    """Load metrics for a specific subtest."""
    metrics_file = eval_results_dir / f"fullduplexbench.{subtest}" / "metrics.json"
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Extract greedy metrics
    subtest_key = f"fullduplexbench.{subtest}"
    if subtest_key in data and "greedy" in data[subtest_key]:
        return data[subtest_key]["greedy"]
    
    return None


def collect_all_metrics(eval_results_dir: Path, subtests: List[str] = None) -> Dict:
    """Collect metrics from all subtests."""
    if subtests is None:
        subtests = SUBTESTS
    
    results = {}
    
    for subtest in subtests:
        metrics = load_metrics(eval_results_dir, subtest)
        if metrics:
            results[subtest] = metrics
        else:
            print(f"⚠️  Warning: No metrics found for {subtest}")
    
    return results


def compute_aggregate_stats(results: Dict) -> Dict:
    """Compute aggregate statistics across subtests."""
    if not results:
        return {}
    
    # Collect all metric names
    all_metric_names = set()
    for metrics in results.values():
        all_metric_names.update(metrics.keys())
    
    aggregate = {}
    
    for metric_name in all_metric_names:
        values = []
        for subtest_metrics in results.values():
            if metric_name in subtest_metrics:
                value = subtest_metrics[metric_name]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if values:
            aggregate[metric_name] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values),
            }
    
    return aggregate


def print_summary(results: Dict, aggregate: Dict):
    """Print formatted summary of results."""
    print("\n" + "="*80)
    print("FULL-DUPLEX-BENCH RESULTS SUMMARY")
    print("="*80)
    
    if not results:
        print("No results found.")
        return
    
    # Print per-subtest results
    print("\nPer-Subtest Results:")
    print("-"*80)
    
    for subtest in SUBTESTS:
        if subtest in results:
            print(f"\n{subtest.upper()}:")
            metrics = results[subtest]
            for key, value in sorted(metrics.items()):
                if isinstance(value, (int, float)):
                    print(f"  {key:30s}: {value:.4f}")
                else:
                    print(f"  {key:30s}: {value}")
        else:
            print(f"\n{subtest.upper()}: No results")
    
    # Print aggregate statistics
    if aggregate:
        print("\n" + "-"*80)
        print("AGGREGATE STATISTICS (across all subtests):")
        print("-"*80)
        
        for metric_name, stats in sorted(aggregate.items()):
            print(f"\n{metric_name}:")
            print(f"  Mean : {stats['mean']:.4f}")
            print(f"  Min  : {stats['min']:.4f}")
            print(f"  Max  : {stats['max']:.4f}")
            print(f"  Count: {stats['count']}")
    
    print("\n" + "="*80 + "\n")


def export_to_csv(results: Dict, output_file: Path):
    """Export results to CSV for analysis."""
    if not results:
        print("No results to export.")
        return
    
    # Prepare data for DataFrame
    rows = []
    for subtest, metrics in results.items():
        row = {'subtest': subtest}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✓ Results exported to: {output_file}")


def export_summary_json(results: Dict, aggregate: Dict, output_file: Path):
    """Export complete summary to JSON."""
    summary = {
        'per_subtest': results,
        'aggregate': aggregate,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Full-Duplex-Bench results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all subtests
  python aggregate_results.py \\
    --eval_results_dir /path/to/eval-results
  
  # Aggregate specific subtests and export to CSV
  python aggregate_results.py \\
    --eval_results_dir /path/to/eval-results \\
    --subtests pause turn_taking \\
    --csv results.csv
  
  # Export summary JSON
  python aggregate_results.py \\
    --eval_results_dir /path/to/eval-results \\
    --json summary.json
        """
    )
    parser.add_argument(
        "--eval_results_dir",
        type=Path,
        required=True,
        help="Path to eval-results directory containing subtest results"
    )
    parser.add_argument(
        "--subtests",
        nargs="+",
        default=None,
        help="Specific subtests to aggregate (default: all)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Export summary to JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not args.eval_results_dir.exists():
        print(f"Error: Directory not found: {args.eval_results_dir}")
        return 1
    
    # Collect metrics
    subtests = args.subtests if args.subtests else SUBTESTS
    print(f"Collecting results for: {', '.join(subtests)}")
    
    results = collect_all_metrics(args.eval_results_dir, subtests)
    
    if not results:
        print("No results found. Make sure scoring has been completed.")
        return 1
    
    # Compute aggregate statistics
    aggregate = compute_aggregate_stats(results)
    
    # Print summary
    print_summary(results, aggregate)
    
    # Export if requested
    if args.csv:
        export_to_csv(results, args.csv)
    
    if args.json:
        export_summary_json(results, aggregate, args.json)
    
    print("✓ Aggregation completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
