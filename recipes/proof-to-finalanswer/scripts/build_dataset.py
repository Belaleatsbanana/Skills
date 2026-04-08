#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("output*.jsonl"))


def _choose_best(existing: dict | None, candidate: dict) -> dict:
    if existing is None:
        return candidate
    existing_score = existing.get("equivalence_score")
    candidate_score = candidate.get("equivalence_score")
    existing_score = -1.0 if existing_score is None else float(existing_score)
    candidate_score = -1.0 if candidate_score is None else float(candidate_score)
    if candidate_score > existing_score:
        return candidate
    return existing


def build_final_dataset(
    input_path: Path,
    output_file: Path,
    required_equivalence_score: float,
    dedup_key: str,
) -> tuple[int, int]:
    best_by_key: dict[str, dict] = {}
    total_rows = 0

    for jsonl_path in _iter_input_files(input_path):
        with open(jsonl_path, "rt", encoding="utf-8") as fin:
            for line in fin:
                total_rows += 1
                row = json.loads(line)

                final_output = row.get("final_output") or {}
                fa_problem = final_output.get("problem", row.get("final_answer_problem"))
                fa_answer = final_output.get("expected_answer", row.get("final_answer_expected_answer"))
                eq_score = row.get("equivalence_score")

                if fa_problem is None or fa_answer is None or eq_score is None:
                    continue
                if float(eq_score) < required_equivalence_score:
                    continue

                original_problem = row.get(dedup_key)
                if original_problem is None:
                    original_problem = row.get("proof_problem")
                if original_problem is None:
                    original_problem = f"row-{total_rows}"

                dataset_row = {
                    "problem": fa_problem,
                    "expected_answer": fa_answer,
                    "source_problem": row.get("proof_problem", row.get(dedup_key)),
                    "equivalence_score": eq_score,
                    "pipeline_status": row.get("pipeline_status"),
                    "metadata": {
                        "proof_reference_solution_source": row.get("proof_reference_solution_source"),
                        "transform_explanation": row.get("transformation", {}).get("equivalency_explanation"),
                        "answer_verification_status": (row.get("answer_verification") or {}).get("status"),
                        "answer_verification_analysis": (row.get("answer_verification") or {}).get("analysis"),
                        "transform_expected_answer": row.get("final_answer_expected_answer"),
                        "solver_predicted_answer": row.get("final_answer_predicted_answer"),
                    },
                }
                best_by_key[str(original_problem)] = _choose_best(best_by_key.get(str(original_problem)), dataset_row)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wt", encoding="utf-8") as fout:
        for row in best_by_key.values():
            fout.write(json.dumps(row) + "\n")

    return total_rows, len(best_by_key)


def main():
    parser = argparse.ArgumentParser(description="Build final-answer dataset from conversion pipeline outputs.")
    parser.add_argument("--input_path", type=Path, required=True, help="Input output*.jsonl file or directory.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output jsonl with successful conversions.")
    parser.add_argument(
        "--required_equivalence_score",
        type=float,
        default=1.0,
        help="Minimum equivalence score to keep (default: 1.0).",
    )
    parser.add_argument(
        "--dedup_key",
        type=str,
        default="problem",
        help="Field used to deduplicate rows by original proof problem.",
    )
    args = parser.parse_args()

    total_rows, kept_rows = build_final_dataset(
        input_path=args.input_path,
        output_file=args.output_file,
        required_equivalence_score=args.required_equivalence_score,
        dedup_key=args.dedup_key,
    )

    print(f"Processed rows: {total_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Saved dataset to: {args.output_file}")


if __name__ == "__main__":
    main()
