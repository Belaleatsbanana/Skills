#!/usr/bin/env python3
"""
After stage 1 (solve-proof-for-quality): copy model generation into 'solution' field
so stage 2 (assess-problem-quality-with-solution) has both problem and solution.
Reads output.jsonl, writes solved.jsonl in the same dir.
"""

import argparse
import json


def _str(x):
    return x if isinstance(x, str) else (str(x) if x else "")


def get_solution(item):
    """Combine reasoning_content + final proof (generation/serialized_output) for stage 2 judge."""
    parts = []

    # 1. Reasoning / chain-of-thought (top-level or inside serialized_output)
    rc = item.get("reasoning_content")
    if rc:
        parts.append(_str(rc))
    serialized = item.get("serialized_output") or []
    if not rc and serialized and len(serialized) > 0 and isinstance(serialized[0], dict):
        rc = serialized[0].get("reasoning_content")
        if rc:
            parts.append(_str(rc))

    # 2. Final proof: generation, or serialized_output[0].content
    gen = item.get("generation")
    if gen:
        parts.append(_str(gen))
    elif serialized and len(serialized) > 0 and isinstance(serialized[0], dict):
        content = serialized[0].get("content") or serialized[0].get("text")
        if content:
            parts.append(_str(content))

    if not parts:
        return ""
    return "\n\n---\n\n".join(p for p in parts if p.strip())


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
            # Stage 2 prompt expects "problem" and "solution"
            if "problem" not in item and "question" in item:
                item["problem"] = item["question"]
            if "problem" not in item and "input" in item:
                item["problem"] = item["input"]
            item["solution"] = get_solution(item)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} rows to {args.output_file}")


if __name__ == "__main__":
    main()
