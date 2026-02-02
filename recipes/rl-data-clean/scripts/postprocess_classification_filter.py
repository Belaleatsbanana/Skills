#!/usr/bin/env python3
"""
Filter problems based on 4 binary classifications:
- Remove: MCQ, Binary questions, Invalid problems
- Separate: Proof vs Non-proof
"""

import argparse
import json


def parse_classification(text: str, expected_positive: str, expected_negative: str) -> bool:
    """
    Parse binary classification output.
    Returns True if positive, False if negative, None if uncertain.
    """
    text_lower = text.lower().strip()

    # Check for positive match
    if expected_positive.lower() in text_lower:
        return True
    # Check for negative match
    elif expected_negative.lower() in text_lower:
        return False
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Filter problems by classification")
    parser.add_argument("input_file", help="Input JSONL with all 4 classifications")
    parser.add_argument("output_proof", help="Output file for proof problems")
    parser.add_argument("output_non_proof", help="Output file for valid non-proof problems")
    parser.add_argument("output_removed", help="Output file for removed problems (MCQ/Binary/Invalid)")

    args = parser.parse_args()

    proof_problems = []
    non_proof_problems = []
    removed_problems = []

    stats = {
        "total": 0,
        "is_proof": 0,
        "is_mcq": 0,
        "is_binary": 0,
        "is_invalid": 0,
        "proof_output": 0,
        "non_proof_output": 0,
        "removed": 0,
        "uncertain": 0,
    }

    with open(args.input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            stats["total"] += 1

            # Get classification results
            is_proof_gen = item.get("is_proof_generation", "")
            is_mcq_gen = item.get("is_mcq_generation", "")
            is_binary_gen = item.get("is_binary_generation", "")
            is_invalid_gen = item.get("is_invalid_generation", "")

            # Parse classifications
            is_proof = parse_classification(is_proof_gen, "proof", "not proof")
            is_mcq = parse_classification(is_mcq_gen, "mcq", "not mcq")
            is_binary = parse_classification(is_binary_gen, "binary", "not binary")
            is_invalid = parse_classification(is_invalid_gen, "invalid", "not invalid")

            # Count classifications
            if is_proof:
                stats["is_proof"] += 1
            if is_mcq:
                stats["is_mcq"] += 1
            if is_binary:
                stats["is_binary"] += 1
            if is_invalid:
                stats["is_invalid"] += 1

            # Add parsed results to item
            item["classifications"] = {
                "is_proof": is_proof,
                "is_mcq": is_mcq,
                "is_binary": is_binary,
                "is_invalid": is_invalid,
            }

            # Decision logic
            # Remove if: MCQ, Binary, or Invalid
            if is_mcq or is_binary or is_invalid:
                removed_problems.append(item)
                item["removal_reason"] = []
                if is_mcq:
                    item["removal_reason"].append("MCQ")
                if is_binary:
                    item["removal_reason"].append("Binary question")
                if is_invalid:
                    item["removal_reason"].append("Invalid problem")
                stats["removed"] += 1
            # If any classification is uncertain, treat as uncertain (optionally remove)
            elif is_proof is None or is_mcq is None or is_binary is None or is_invalid is None:
                stats["uncertain"] += 1
                # Conservatively remove uncertain cases
                removed_problems.append(item)
                item["removal_reason"] = ["Uncertain classification"]
                stats["removed"] += 1
            # Valid problems: separate by proof vs non-proof
            elif is_proof:
                proof_problems.append(item)
                stats["proof_output"] += 1
            else:
                non_proof_problems.append(item)
                stats["non_proof_output"] += 1

    # Write outputs
    with open(args.output_proof, "w") as f:
        for item in proof_problems:
            f.write(json.dumps(item) + "\n")

    with open(args.output_non_proof, "w") as f:
        for item in non_proof_problems:
            f.write(json.dumps(item) + "\n")

    with open(args.output_removed, "w") as f:
        for item in removed_problems:
            f.write(json.dumps(item) + "\n")

    # Print statistics
    print(f"\n{'=' * 70}")
    print("Classification Filtering Results")
    print(f"{'=' * 70}")
    print(f"Total items processed: {stats['total']}")
    print("\nClassification counts:")
    print(f"  Proof problems:     {stats['is_proof']:4d} ({stats['is_proof'] / stats['total'] * 100:.1f}%)")
    print(f"  MCQ:                {stats['is_mcq']:4d} ({stats['is_mcq'] / stats['total'] * 100:.1f}%)")
    print(f"  Binary questions:   {stats['is_binary']:4d} ({stats['is_binary'] / stats['total'] * 100:.1f}%)")
    print(f"  Invalid:            {stats['is_invalid']:4d} ({stats['is_invalid'] / stats['total'] * 100:.1f}%)")
    print(f"  Uncertain:          {stats['uncertain']:4d} ({stats['uncertain'] / stats['total'] * 100:.1f}%)")
    print("\nOutput distribution:")
    print(f"  Proof problems:     {stats['proof_output']:4d} ({stats['proof_output'] / stats['total'] * 100:.1f}%)")
    print(
        f"  Non-proof problems: {stats['non_proof_output']:4d} ({stats['non_proof_output'] / stats['total'] * 100:.1f}%)"
    )
    print(f"  Removed:            {stats['removed']:4d} ({stats['removed'] / stats['total'] * 100:.1f}%)")
    print(f"{'=' * 70}\n")

    print(f"Proof problems:     {args.output_proof}")
    print(f"Non-proof problems: {args.output_non_proof}")
    print(f"Removed problems:   {args.output_removed}")


if __name__ == "__main__":
    main()
