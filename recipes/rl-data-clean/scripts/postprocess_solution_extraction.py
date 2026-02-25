#!/usr/bin/env python3
"""
Postprocess solution extraction outputs for non-proof problems.
Extracts clean solution from model generation.
"""

import argparse
import json
import re


def extract_final_answer(text: str):
    """
    Extract only the final answer (e.g. "18", "41", "\\sqrt{2}") from solution text.
    Prefers **Answer:** or Answer: or \\boxed{...}; returns last occurrence so we get the final answer.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    candidates = []

    # Last **Answer:** ... (capture until next ** or double newline or end)
    for m in re.finditer(r"\*\*Answer:\*\*\s*(.+?)(?=\n\n|\*\*|$)", text, re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())
    # Last Answer: ... (same)
    for m in re.finditer(r"(?<!\*)\bAnswer:\s*(.+?)(?=\n\n|\*\*|$)", text, re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())
    # Last \boxed{...}
    for m in re.finditer(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text):
        candidates.append(m.group(1).strip())

    if candidates:
        # Take last (final) answer, clean to one line if possible
        ans = candidates[-1]
        ans = re.sub(r"\s+", " ", ans).strip()
        ans = re.sub(r"\*+\.?\s*$", "", ans).strip()  # trailing ** or .**
        if len(ans) > 500:
            ans = ans[:500] + "…"  # cap very long answers
        return ans if ans else None
    return None


def extract_solution(generation: str) -> str:
    """
    Extract solution from model generation.
    Returns full solution text; use extract_final_answer() for expected_answer.
    """
    if not generation:
        return None

    # Check for "Solution not found"
    if "solution not found" in generation.lower():
        return None

    # Try to extract everything after "**Solution:**" or "Solution:"
    patterns = [
        r"\*\*Solution:\*\*\s*(.+)",
        r"Solution:\s*(.+)",
        r"Output:\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, generation, re.IGNORECASE | re.DOTALL)
        if match:
            solution = match.group(1).strip()
            # Remove trailing markers
            solution = re.sub(r"\n\s*Output:\s*$", "", solution, flags=re.IGNORECASE)
            if solution and solution != "Solution not found.":
                return solution

    # If no pattern matched, return the whole generation if it's substantial
    if len(generation.strip()) > 50:
        return generation.strip()

    return None


def main():
    parser = argparse.ArgumentParser(description="Extract solutions from generation output")
    parser.add_argument("input_file", help="Input JSONL with generation field")
    parser.add_argument("output_file", help="Output JSONL with extracted solution")

    args = parser.parse_args()

    extracted = []
    not_found = 0
    total = 0

    with open(args.input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            total += 1

            # Get generation
            generation = item.get("generation", "")
            if not generation and "serialized_output" in item:
                serialized = item.get("serialized_output", [])
                if serialized and len(serialized) > 0:
                    generation = serialized[0].get("content", "")

            # Extract solution (full text) and expected_answer (final answer only)
            solution = extract_solution(generation)
            if solution:
                # Prefer short final answer for expected_answer; fallback to full solution
                expected = extract_final_answer(generation) or extract_final_answer(solution) or solution
                item["extracted_solution"] = solution
                item["expected_answer"] = expected
                item["solution_extraction_gen"] = generation
                extracted.append(item)
            else:
                not_found += 1

    # Write output
    with open(args.output_file, "w") as f:
        for item in extracted:
            f.write(json.dumps(item) + "\n")

    # Print statistics
    print(f"\n{'=' * 70}")
    print("Solution Extraction Results")
    print(f"{'=' * 70}")
    print(f"Total items:          {total}")
    print(f"Solutions extracted:  {len(extracted)} ({len(extracted) / total * 100:.1f}%)")
    print(f"Solutions not found:  {not_found} ({not_found / total * 100:.1f}%)")
    print(f"{'=' * 70}\n")

    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
