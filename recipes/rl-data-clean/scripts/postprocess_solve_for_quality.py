#!/usr/bin/env python3
"""
After stage 1 (solve-proof-for-quality): copy model generation into 'solution' field
so stage 2 (assess-problem-quality-with-solution) has both problem and solution.
Reads output.jsonl, writes solved.jsonl in the same dir.
"""

import argparse
import json


def get_solution(item):
    """Extract solution text from generation, serialized_output, or reasoning_content."""
    gen = item.get("generation") or ""
    if gen and isinstance(gen, str):
        return gen
    if item.get("serialized_output"):
        serialized = item.get("serialized_output", [])
        if serialized and len(serialized) > 0:
            content = serialized[0].get("content") or serialized[0].get("text")
            if content:
                return content if isinstance(content, str) else str(content)
    if item.get("reasoning_content"):
        rc = item.get("reasoning_content")
        return rc if isinstance(rc, str) else str(rc)
    return ""


def main():
    parser = argparse.ArgumentParser(description="Add solution field from generation for stage 2 input")
    parser.add_argument("input_file", help="Stage 1 output.jsonl")
    parser.add_argument("output_file", help="Output solved.jsonl with problem + solution")
    args = parser.parse_args()

    count = 0
    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            item["solution"] = get_solution(item)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} rows to {args.output_file}")


if __name__ == "__main__":
    main()
