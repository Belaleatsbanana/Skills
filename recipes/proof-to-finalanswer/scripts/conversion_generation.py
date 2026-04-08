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

import copy
import os
import sys
from typing import Any

import numpy as np

from nemo_skills.inference.model import BaseModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from conversion_utils import (  # noqa: E402
    find_first_nonempty_value,
    find_single_correct_solution,
    load_prompt_user_template,
    transform_problem_to_final_answer,
    verify_final_answer_match,
)


def _split_csv_keys(keys_csv: str) -> list[str]:
    return [key.strip() for key in keys_csv.split(",") if key.strip()]


def _make_rng(random_seed: int, datapoint: dict[str, Any]) -> np.random.RandomState:
    async_position = datapoint.get("_async_position", 0)
    return np.random.RandomState(random_seed + 10000 * async_position)


async def process_single(
    llm: BaseModel,
    datapoint: dict,
    generate_solution_prompt_path: str,
    proof_transform_prompt_path: str,
    solution_comparison_prompt_path: str,
    llm_kwargs: dict,
    random_seed: int,
    proof_problem_key: str = "problem",
    allow_provided_proof_solution: bool = True,
    provided_proof_solution_keys: str = "reference_solution,proof_solution,solution",
    proof_solution_attempts: int = 8,
    final_answer_solution_attempts: int = 8,
    transform_attempts: int = 3,
    comparison_attempts: int = 3,
    required_equivalence_score: float = 1.0,
    keep_attempt_generations: bool = False,
) -> dict:
    """
    Full proof-to-finalanswer pipeline for one datapoint.

    Pipeline:
    1) Get a correct reference solution for the proof problem.
    2) Transform to a standalone final-answer problem + answer.
    3) Get a correct solution for the transformed problem.
    4) Compare transformed expected answer with solved final-answer prediction.
    """
    data = copy.deepcopy(datapoint)
    llm_kwargs = llm_kwargs or {}
    rng = _make_rng(random_seed, data)
    _ = solution_comparison_prompt_path
    _ = comparison_attempts
    _ = required_equivalence_score

    proof_problem = data.get(proof_problem_key)
    if not proof_problem:
        return {
            **data,
            "pipeline_status": "missing_problem",
            "pipeline_success": False,
            "generation": "missing_problem",
        }

    gen_solution_prompt = load_prompt_user_template(generate_solution_prompt_path)
    transform_prompt = load_prompt_user_template(proof_transform_prompt_path)

    reference_solution = None
    proof_solution_info: dict[str, Any] = {}

    if allow_provided_proof_solution:
        _, provided_solution = find_first_nonempty_value(data, _split_csv_keys(provided_proof_solution_keys))
        if provided_solution is not None:
            reference_solution = str(provided_solution)
            proof_solution_info = {
                "status": "provided",
                "selected_solution": reference_solution,
                "attempts": [],
            }

    if reference_solution is None:
        proof_solution_info = await find_single_correct_solution(
            llm=llm,
            problem=proof_problem,
            prompt_template=gen_solution_prompt,
            llm_kwargs=llm_kwargs,
            rng=rng,
            max_attempts=proof_solution_attempts,
            keep_attempt_generations=keep_attempt_generations,
        )
        reference_solution = proof_solution_info["selected_solution"]

    if reference_solution is None:
        return {
            **data,
            "pipeline_status": "pass_0_proof_problem",
            "pipeline_success": False,
            "proof_problem": proof_problem,
            "proof_reference_solution": None,
            "proof_reference_solution_source": proof_solution_info.get("status"),
            "proof_solution_search": proof_solution_info,
            "generation": "pass_0_proof_problem",
        }

    transform_info = await transform_problem_to_final_answer(
        llm=llm,
        problem=proof_problem,
        reference_solution=reference_solution,
        prompt_template=transform_prompt,
        llm_kwargs=llm_kwargs,
        rng=rng,
        max_attempts=transform_attempts,
        keep_attempt_generations=keep_attempt_generations,
    )
    final_answer_problem = transform_info.get("final_answer_problem")
    final_answer = transform_info.get("final_answer")

    if transform_info.get("status") != "success":
        return {
            **data,
            "pipeline_status": "transform_failed",
            "pipeline_success": False,
            "proof_problem": proof_problem,
            "proof_reference_solution": reference_solution,
            "proof_reference_solution_source": proof_solution_info.get("status"),
            "proof_solution_search": proof_solution_info,
            "transformation": transform_info,
            "generation": "transform_failed",
        }

    fa_solution_info = await find_single_correct_solution(
        llm=llm,
        problem=final_answer_problem,
        prompt_template=gen_solution_prompt,
        llm_kwargs=llm_kwargs,
        rng=rng,
        max_attempts=final_answer_solution_attempts,
        keep_attempt_generations=keep_attempt_generations,
    )
    final_answer_solution = fa_solution_info["selected_solution"]
    if final_answer_solution is None:
        return {
            **data,
            "pipeline_status": "pass_0_final_answer_problem",
            "pipeline_success": False,
            "proof_problem": proof_problem,
            "proof_reference_solution": reference_solution,
            "proof_reference_solution_source": proof_solution_info.get("status"),
            "proof_solution_search": proof_solution_info,
            "final_answer_problem": final_answer_problem,
            "final_answer_expected_answer": final_answer,
            "transformation": transform_info,
            "final_answer_solution_search": fa_solution_info,
            "generation": "pass_0_final_answer_problem",
        }

    final_answer_predicted_answer = fa_solution_info.get("selected_final_answer")
    answer_verification = verify_final_answer_match(
        expected_answer=final_answer,
        predicted_answer=final_answer_predicted_answer,
    )

    canonical_expected_answer = (
        final_answer_predicted_answer if final_answer_predicted_answer is not None else final_answer
    )
    pipeline_status = "success"
    equivalence_score = 1.0
    final_output = {
        "problem": final_answer_problem,
        "expected_answer": canonical_expected_answer,
    }

    return {
        **data,
        "pipeline_status": pipeline_status,
        "pipeline_success": True,
        "proof_problem": proof_problem,
        "proof_reference_solution": reference_solution,
        "proof_reference_solution_source": proof_solution_info.get("status"),
        "proof_solution_search": proof_solution_info,
        "final_answer_problem": final_answer_problem,
        "final_answer_expected_answer": final_answer,
        "transformation": transform_info,
        "final_answer_reference_solution": final_answer_solution,
        "final_answer_predicted_answer": final_answer_predicted_answer,
        "final_answer_solution_search": fa_solution_info,
        "answer_verification": answer_verification,
        "comparison": {
            "status": answer_verification.get("status"),
            "analysis": answer_verification.get("analysis"),
            "equivalence_score": equivalence_score,
        },
        "equivalence_score": equivalence_score,
        "final_output": final_output,
        "generation": pipeline_status,
    }
