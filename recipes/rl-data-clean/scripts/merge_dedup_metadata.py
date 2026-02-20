#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Merge metadata back into dedup.jsonl: each line in dedup has (problem, similar_items).
# We look up the full row for that problem from a source jsonl (e.g. filtered.jsonl)
# and output source row + similar_items so downstream stages keep all metadata.
#
# Usage:
#   python merge_dedup_metadata.py --source filtered.jsonl --dedup dedup.jsonl --output dedup_with_metadata.jsonl

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="Merge full-row metadata from source jsonl into dedup.jsonl")
    ap.add_argument(
        "--source", required=True, help="Source jsonl (e.g. filtered.jsonl) with full rows keyed by problem"
    )
    ap.add_argument(
        "--dedup", required=True, help="dedup.jsonl from cluster_by_contamination (problem, similar_items)"
    )
    ap.add_argument("--output", required=True, help="Output jsonl: full row per representative + similar_items")
    ap.add_argument("--key", default="problem", help="Key to join on (default: problem)")
    args = ap.parse_args()

    # problem -> first full row from source
    key_to_row = {}
    with open(args.source, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            k = row.get(args.key)
            if k is not None and k not in key_to_row:
                key_to_row[k] = row.copy()

    n_out = 0
    n_miss = 0
    with open(args.dedup, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            dedup_row = json.loads(line)
            k = dedup_row.get(args.key)
            similar_items = dedup_row.get("similar_items", [])
            if k is None:
                n_miss += 1
                out = {"problem": None, "similar_items": similar_items}
            else:
                base = key_to_row.get(k)
                if base is None:
                    n_miss += 1
                    out = {"problem": k, "similar_items": similar_items}
                else:
                    out = {**base, "similar_items": similar_items}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    if n_miss > 0:
        print(f"Warning: {n_miss} dedup rows had no matching key in source", file=sys.stderr)
    print(f"Wrote {n_out} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
