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
Compare Full-Duplex-Bench evaluation results across multiple runs/models.

This script helps analyze:
1. Performance differences between models
2. Improvements across training iterations
3. Impact of different configurations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


SUBTESTS = ["pause", "backchannel", "turn_taking", "interruption"]


def load_run_metrics(eval_results_dir: Path, run_name: str, subtests: List[str]) -> Dict:
    """Load metrics for all subtests in a run."""
    run_results = {'run_name': run_name}
    
    for subtest in subtests:
        metrics_file = eval_results_dir / f"fullduplexbench.{subtest}" / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            subtest_key = f"fullduplexbench.{subtest}"
            if subtest_key in data and "greedy" in data[subtest_key]:
                metrics = data[subtest_key]["greedy"]
                for metric_name, value in metrics.items():
                    key = f"{subtest}_{metric_name}"
                    run_results[key] = value
    
    return run_results


def compare_runs(runs: List[Dict], baseline_run: str = None) -> pd.DataFrame:
    """Compare multiple runs and compute differences."""
    df = pd.DataFrame(runs)
    
    if baseline_run and baseline_run in df['run_name'].values:
        baseline_idx = df[df['run_name'] == baseline_run].index[0]
        baseline_row = df.iloc[baseline_idx]
        
        # Add delta columns
        for col in df.columns:
            if col != 'run_name' and pd.api.types.is_numeric_dtype(df[col]):
                baseline_val = baseline_row[col]
                if pd.notna(baseline_val) and baseline_val != 0:
                    df[f'{col}_delta'] = df[col] - baseline_val
                    df[f'{col}_delta_pct'] = ((df[col] - baseline_val) / baseline_val) * 100
    
    return df


def print_comparison(df: pd.DataFrame, metric_filter: str = None):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("FULL-DUPLEX-BENCH COMPARISON")
    print("="*100)
    
    # Display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    if metric_filter:
        # Filter columns by metric name
        cols_to_show = ['run_name']
        for col in df.columns:
            if metric_filter.lower() in col.lower():
                cols_to_show.append(col)
        
        if len(cols_to_show) > 1:
            print(f"\nFiltered metrics containing '{metric_filter}':")
            print("-"*100)
            print(df[cols_to_show].to_string(index=False))
        else:
            print(f"\n⚠️  No metrics found matching '{metric_filter}'")
    else:
        # Show all metrics by subtest
        for subtest in SUBTESTS:
            subtest_cols = ['run_name']
            for col in df.columns:
                if col.startswith(f"{subtest}_") and not col.endswith('_delta') and not col.endswith('_delta_pct'):
                    subtest_cols.append(col)
                    # Add delta columns if they exist
                    delta_col = f"{col}_delta"
                    delta_pct_col = f"{col}_delta_pct"
                    if delta_col in df.columns:
                        subtest_cols.append(delta_col)
                    if delta_pct_col in df.columns:
                        subtest_cols.append(delta_pct_col)
            
            if len(subtest_cols) > 1:
                print(f"\n{subtest.upper()}:")
                print("-"*100)
                print(df[subtest_cols].to_string(index=False))
    
    print("\n" + "="*100 + "\n")


def export_comparison(df: pd.DataFrame, output_file: Path, format: str = 'csv'):
    """Export comparison to file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(output_file, index=False)
    elif format == 'excel':
        df.to_excel(output_file, index=False, engine='openpyxl')
    elif format == 'json':
        df.to_json(output_file, orient='records', indent=2)
    
    print(f"✓ Comparison exported to: {output_file}")


def generate_report(df: pd.DataFrame, output_file: Path, baseline_run: str = None):
    """Generate a detailed markdown report."""
    with open(output_file, 'w') as f:
        f.write("# Full-Duplex-Bench Evaluation Comparison\n\n")
        
        if baseline_run:
            f.write(f"**Baseline Run:** `{baseline_run}`\n\n")
        
        f.write(f"**Number of Runs:** {len(df)}\n\n")
        f.write(f"**Runs Compared:**\n")
        for run_name in df['run_name']:
            f.write(f"- {run_name}\n")
        f.write("\n")
        
        # Summary by subtest
        for subtest in SUBTESTS:
            f.write(f"## {subtest.upper()}\n\n")
            
            subtest_cols = ['run_name']
            for col in df.columns:
                if col.startswith(f"{subtest}_") and not col.endswith('_delta') and not col.endswith('_delta_pct'):
                    subtest_cols.append(col)
            
            if len(subtest_cols) > 1:
                subtest_df = df[subtest_cols]
                f.write(subtest_df.to_markdown(index=False))
                f.write("\n\n")
                
                # Highlight best performance
                for col in subtest_cols[1:]:  # Skip run_name
                    if pd.api.types.is_numeric_dtype(df[col]):
                        best_idx = df[col].idxmax()
                        best_run = df.loc[best_idx, 'run_name']
                        best_val = df.loc[best_idx, col]
                        f.write(f"- **Best {col}:** {best_run} ({best_val:.4f})\n")
                f.write("\n")
        
        # Overall statistics
        f.write("## Overall Statistics\n\n")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            f.write(stats.to_markdown())
            f.write("\n")
    
    print(f"✓ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Full-Duplex-Bench results across multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs
  python compare_eval_results.py \\
    --runs run1:/path/to/eval-results1 run2:/path/to/eval-results2
  
  # Compare with baseline
  python compare_eval_results.py \\
    --runs baseline:/path/to/baseline improved:/path/to/improved \\
    --baseline baseline
  
  # Export comparison and generate report
  python compare_eval_results.py \\
    --runs model_v1:/path/to/v1 model_v2:/path/to/v2 \\
    --output comparison.csv \\
    --report comparison_report.md
  
  # Filter specific metric
  python compare_eval_results.py \\
    --runs run1:/path/to/eval-results1 run2:/path/to/eval-results2 \\
    --metric takeover_rate
        """
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specifications as 'name:/path/to/eval-results' pairs"
    )
    parser.add_argument(
        "--subtests",
        nargs="+",
        default=SUBTESTS,
        help="Subtests to compare (default: all)"
    )
    parser.add_argument(
        "--baseline",
        help="Name of baseline run for computing deltas"
    )
    parser.add_argument(
        "--metric",
        help="Filter display by metric name (e.g., 'takeover_rate')"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Export comparison to file (format based on extension: .csv, .xlsx, .json)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Generate detailed markdown report"
    )
    
    args = parser.parse_args()
    
    # Parse runs
    runs_data = []
    for run_spec in args.runs:
        try:
            run_name, run_path = run_spec.split(':', 1)
            run_path = Path(run_path)
            
            if not run_path.exists():
                print(f"⚠️  Warning: Path not found: {run_path}")
                continue
            
            run_metrics = load_run_metrics(run_path, run_name, args.subtests)
            runs_data.append(run_metrics)
            
        except ValueError:
            print(f"⚠️  Warning: Invalid run spec format: {run_spec}")
            print("    Expected format: 'name:/path/to/eval-results'")
    
    if not runs_data:
        print("Error: No valid runs found.")
        return 1
    
    # Compare runs
    comparison_df = compare_runs(runs_data, baseline_run=args.baseline)
    
    # Print comparison
    print_comparison(comparison_df, metric_filter=args.metric)
    
    # Export if requested
    if args.output:
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.json': 'json',
        }
        file_format = format_map.get(args.output.suffix, 'csv')
        export_comparison(comparison_df, args.output, format=file_format)
    
    # Generate report if requested
    if args.report:
        generate_report(comparison_df, args.report, baseline_run=args.baseline)
    
    print("✓ Comparison completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
