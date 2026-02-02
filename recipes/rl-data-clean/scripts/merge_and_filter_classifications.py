#!/usr/bin/env python3
"""
Merge 4 classification outputs into a single file, then filter.

This script:
1. Reads extracted-problems.jsonl (base file)
2. Joins it with 4 classification outputs
3. Filters into proof/non-proof/removed
"""

import argparse
import json
from pathlib import Path


def load_jsonl(filepath):
    """Load JSONL file into a dict keyed by problem index or unique ID."""
    data = {}
    with open(filepath, "r") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            # Use extracted_problem_idx if available, otherwise use line number
            key = item.get("extracted_problem_idx", idx)
            data[key] = item
    return data


def merge_classifications(base_file, proof_file, mcq_file, binary_file, invalid_file):
    """Merge all classification results into base data."""
    # Load base data
    base_data = load_jsonl(base_file)

    # Load classifications
    proof_data = load_jsonl(proof_file)
    mcq_data = load_jsonl(mcq_file)
    binary_data = load_jsonl(binary_file)
    invalid_data = load_jsonl(invalid_file)

    # Merge
    merged = []
    for key, item in base_data.items():
        # Add classifications
        if key in proof_data:
            item["is_proof_generation"] = proof_data[key].get("is_proof_generation", "")
        if key in mcq_data:
            item["is_mcq_generation"] = mcq_data[key].get("is_mcq_generation", "")
        if key in binary_data:
            item["is_binary_generation"] = binary_data[key].get("is_binary_generation", "")
        if key in invalid_data:
            item["is_invalid_generation"] = invalid_data[key].get("is_invalid_generation", "")

        merged.append(item)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge and filter classifications")
    parser.add_argument("--base-file", required=True, help="Base file (extracted-problems.jsonl)")
    parser.add_argument("--proof-file", required=True, help="Proof classification output")
    parser.add_argument("--mcq-file", required=True, help="MCQ classification output")
    parser.add_argument("--binary-file", required=True, help="Binary classification output")
    parser.add_argument("--invalid-file", required=True, help="Invalid classification output")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge classifications
    print("Merging classification outputs...")
    merged = merge_classifications(
        args.base_file,
        args.proof_file,
        args.mcq_file,
        args.binary_file,
        args.invalid_file,
    )

    # Write merged file
    merged_file = output_dir / "merged-classifications.jsonl"
    with open(merged_file, "w") as f:
        for item in merged:
            f.write(json.dumps(item) + "\n")

    print(f"Merged {len(merged)} items")
    print(f"Merged file: {merged_file}")

    # Now run the filter
    print("\nFiltering...")

    proof_output = output_dir / "proof.jsonl"
    non_proof_output = output_dir / "non-proof.jsonl"
    removed_output = output_dir / "removed.jsonl"

    # Import and run the filter function
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from postprocess_classification_filter import main as filter_main

    # Call filter
    sys.argv = [
        "filter",
        str(merged_file),
        str(proof_output),
        str(non_proof_output),
        str(removed_output),
    ]
    filter_main()

    print("\nDone!")
    print(f"Proof problems:     {proof_output}")
    print(f"Non-proof problems: {non_proof_output}")
    print(f"Removed problems:   {removed_output}")


if __name__ == "__main__":
    main()
