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

from typing import Any

import numpy as np
from omegaconf import OmegaConf

from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.inference.model import BaseModel

GENERATE_SOLUTION_SOL_HEADER = "## Solution"
GENERATE_SOLUTION_EVAL_HEADER = "## Self Evaluation"
TRANSFORM_PROBLEM_HEADER = "## Final Answer Problem Statement"
TRANSFORM_EXPLANATION_HEADER = "## EQUIVALENCY EXPLANATION"
TRANSFORM_FINAL_ANSWER_MARKER = "The unique final answer for the problem statement above is:"
COMPARISON_ANALYSIS_PREFIX = "Here is the analysis of the two problem, solution pairs:"
COMPARISON_SCORE_PREFIX = (
    "Based on my evaluation of the two problem, solution pairs, the final equivalence score should be:"
)
VALID_EQUIVALENCE_SCORES = {0.0, 0.5, 1.0}


def load_prompt_user_template(prompt_config_path: str) -> str:
    """Load a prompt config and return its `user` template."""
    prompt_config = OmegaConf.load(prompt_config_path)
    user_prompt = getattr(prompt_config, "user", None)
    if not user_prompt:
        raise ValueError(f"Prompt config at {prompt_config_path} must define a non-empty `user` field.")
    return str(user_prompt)


def remove_thinking_trace(text: str | None) -> str:
    """Drop internal chain-of-thought style prefix from a generation."""
    if text is None:
        return ""
    return text.split("</think>")[-1].strip()


def _find_header_line(lines: list[str], header_prefix: str) -> int | None:
    prefix = header_prefix.strip().lower()
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith(prefix):
            return idx
    return None


def extract_markdown_section(text: str, header_prefix: str) -> str | None:
    """Extract the markdown section that starts at `header_prefix` and ends at the next `##` header."""
    lines = text.splitlines()
    start_idx = _find_header_line(lines, header_prefix)
    if start_idx is None:
        return None
    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip().startswith("## "):
            end_idx = idx
            break
    section = "\n".join(lines[start_idx + 1 : end_idx]).strip()
    return section or None


def _extract_boxed_from_text(text: str) -> str | None:
    return extract_answer(text, extract_from_boxed=True)


def normalize_equivalence_score(raw_score: str | None) -> float | None:
    """Normalize score text into one of {0, 0.5, 1}."""
    if raw_score is None:
        return None
    normalized = raw_score.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")

    frac_forms = {"1/2", "\\frac{1}{2}", "\\tfrac{1}{2}", "{1\\over2}"}
    if normalized in frac_forms:
        return 0.5

    direct_map = {
        "0": 0.0,
        "0.0": 0.0,
        "0.5": 0.5,
        ".5": 0.5,
        "1": 1.0,
        "1.0": 1.0,
    }
    if normalized in direct_map:
        return direct_map[normalized]

    try:
        score = float(normalized)
    except ValueError:
        return None

    if abs(score - 0.0) < 1e-9:
        return 0.0
    if abs(score - 0.5) < 1e-9:
        return 0.5
    if abs(score - 1.0) < 1e-9:
        return 1.0
    return None


def parse_generate_solution_response(response_text: str) -> dict[str, Any]:
    """Parse solution-generation response into solution/evaluation/score fields."""
    clean_text = remove_thinking_trace(response_text)
    boxed_score = _extract_boxed_from_text(clean_text)
    score = normalize_equivalence_score(boxed_score)
    solution_section = extract_markdown_section(clean_text, GENERATE_SOLUTION_SOL_HEADER)
    self_eval_section = extract_markdown_section(clean_text, GENERATE_SOLUTION_EVAL_HEADER)
    parsed_solution = solution_section or clean_text
    return {
        "clean_generation": response_text,
        "solution": parsed_solution,
        "self_evaluation": self_eval_section,
        "self_score_raw": boxed_score,
        "self_score": score,
        "is_correct": score == 1.0,
    }


def _extract_transform_problem_section(transform_text: str) -> str | None:
    """Extract final-answer problem statement from transform output."""
    lines = transform_text.splitlines()
    header_idx = _find_header_line(lines, TRANSFORM_PROBLEM_HEADER)
    if header_idx is None:
        return None
    end_idx = len(lines)
    for idx in range(header_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("## ") or stripped.startswith(TRANSFORM_FINAL_ANSWER_MARKER):
            end_idx = idx
            break
    section = "\n".join(lines[header_idx + 1 : end_idx]).strip()
    return section or None


def parse_proof_transform_response(response_text: str) -> dict[str, Any]:
    """Parse transformed final-answer problem, expected answer, and explanation."""
    clean_text = remove_thinking_trace(response_text)
    final_answer_problem = _extract_transform_problem_section(clean_text)
    explanation_section = extract_markdown_section(clean_text, TRANSFORM_EXPLANATION_HEADER)

    answer_scope = clean_text
    marker_idx = clean_text.find(TRANSFORM_FINAL_ANSWER_MARKER)
    if marker_idx >= 0:
        answer_scope = clean_text[marker_idx:]
    final_answer = _extract_boxed_from_text(answer_scope)

    return {
        "clean_generation": clean_text,
        "final_answer_problem": final_answer_problem,
        "final_answer": final_answer,
        "equivalency_explanation": explanation_section,
        "is_valid": bool(final_answer_problem and final_answer and explanation_section),
    }


def parse_solution_comparison_response(response_text: str) -> dict[str, Any]:
    """Parse comparison explanation and equivalence score."""
    clean_text = remove_thinking_trace(response_text)
    boxed_score = _extract_boxed_from_text(clean_text)
    score = normalize_equivalence_score(boxed_score)

    analysis = None
    start_idx = clean_text.find(COMPARISON_ANALYSIS_PREFIX)
    if start_idx >= 0:
        analysis = clean_text[start_idx:]
        cutoff_idx = analysis.find(COMPARISON_SCORE_PREFIX)
        if cutoff_idx >= 0:
            analysis = analysis[:cutoff_idx].strip()

    return {
        "clean_generation": clean_text,
        "analysis": analysis,
        "equivalence_score_raw": boxed_score,
        "equivalence_score": score,
        "is_valid": score in VALID_EQUIVALENCE_SCORES,
    }


def find_first_nonempty_value(datapoint: dict[str, Any], keys: list[str]) -> tuple[str | None, Any]:
    """Return (key, value) for the first non-empty key among candidates."""
    for key in keys:
        if key in datapoint and datapoint[key] is not None:
            if isinstance(datapoint[key], str) and not datapoint[key].strip():
                continue
            return key, datapoint[key]
    return None, None


def should_accept_equivalence_score(score: float | None, required_min_score: float) -> bool:
    if score is None:
        return False
    return score >= required_min_score


async def llm_call_once(
    llm: BaseModel,
    prompt_text: str,
    llm_kwargs: dict[str, Any],
    req_seed: int,
) -> str:
    """Single LLM call returning cleaned generation text."""
    response = await llm.generate_async(
        prompt=[{"role": "user", "content": prompt_text}],
        **llm_kwargs,
        random_seed=req_seed,
    )
    generation = response.get("generation", "")
    return remove_thinking_trace(generation)


async def find_single_correct_solution(
    llm: BaseModel,
    problem: str,
    prompt_template: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    max_attempts: int,
    keep_attempt_generations: bool,
) -> dict[str, Any]:
    """
    Function 1:
    Generate up to `max_attempts` candidate solutions and return first score-1 solution.
    """
    prompt = prompt_template.format(problem=problem)
    attempts: list[dict[str, Any]] = []
    selected_solution: str | None = None
    selected_idx: int | None = None

    for attempt_idx in range(max_attempts):
        req_seed = int(rng.randint(0, 2**32 - 1))
        generation = await llm_call_once(
            llm=llm,
            prompt_text=prompt,
            llm_kwargs=llm_kwargs,
            req_seed=req_seed,
        )
        parsed = parse_generate_solution_response(generation)
        attempt = {
            "attempt_idx": attempt_idx,
            "request_seed": req_seed,
            "self_score": parsed["self_score"],
            "self_score_raw": parsed["self_score_raw"],
            "is_correct": parsed["is_correct"],
            "solution": parsed["solution"],
        }
        if keep_attempt_generations:
            attempt["generation"] = parsed["clean_generation"]
            attempt["self_evaluation"] = parsed["self_evaluation"]
        attempts.append(attempt)

        if parsed["is_correct"]:
            selected_solution = parsed["solution"]
            selected_idx = attempt_idx
            break

    if selected_solution is None:
        return {
            "status": "pass_0",
            "selected_solution": None,
            "selected_attempt_idx": None,
            "attempts": attempts,
        }
    return {
        "status": "success",
        "selected_solution": selected_solution,
        "selected_attempt_idx": selected_idx,
        "attempts": attempts,
    }


async def transform_problem_to_final_answer(
    llm: BaseModel,
    problem: str,
    reference_solution: str,
    prompt_template: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    max_attempts: int,
    keep_attempt_generations: bool,
) -> dict[str, Any]:
    """
    Function 2:
    Transform proof problem + reference solution into final-answer problem + equivalency explanation.
    """
    prompt = prompt_template.format(problem=problem, solution=reference_solution)
    attempts: list[dict[str, Any]] = []

    for attempt_idx in range(max_attempts):
        req_seed = int(rng.randint(0, 2**32 - 1))
        generation = await llm_call_once(
            llm=llm,
            prompt_text=prompt,
            llm_kwargs=llm_kwargs,
            req_seed=req_seed,
        )
        parsed = parse_proof_transform_response(generation)
        attempt = {
            "attempt_idx": attempt_idx,
            "request_seed": req_seed,
            "is_valid": parsed["is_valid"],
            "final_answer_problem": parsed["final_answer_problem"],
            "final_answer": parsed["final_answer"],
            "equivalency_explanation": parsed["equivalency_explanation"],
        }
        if keep_attempt_generations:
            attempt["generation"] = parsed["clean_generation"]
        attempts.append(attempt)
        if parsed["is_valid"]:
            return {
                "status": "success",
                "selected_attempt_idx": attempt_idx,
                "final_answer_problem": parsed["final_answer_problem"],
                "final_answer": parsed["final_answer"],
                "equivalency_explanation": parsed["equivalency_explanation"],
                "attempts": attempts,
            }

    return {
        "status": "failed",
        "selected_attempt_idx": None,
        "final_answer_problem": None,
        "final_answer": None,
        "equivalency_explanation": None,
        "attempts": attempts,
    }


async def compare_problem_solution_pairs(
    llm: BaseModel,
    proof_problem: str,
    proof_solution: str,
    fa_problem: str,
    fa_solution: str,
    prompt_template: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    max_attempts: int,
    keep_attempt_generations: bool,
) -> dict[str, Any]:
    """
    Function 3:
    Compare the two (problem, solution) pairs and return equivalence score in {0, 0.5, 1}.
    """
    prompt = prompt_template.format(
        proof_problem=proof_problem,
        proof_solution=proof_solution,
        fa_problem=fa_problem,
        fa_solution=fa_solution,
    )
    attempts: list[dict[str, Any]] = []

    for attempt_idx in range(max_attempts):
        req_seed = int(rng.randint(0, 2**32 - 1))
        generation = await llm_call_once(
            llm=llm,
            prompt_text=prompt,
            llm_kwargs=llm_kwargs,
            req_seed=req_seed,
        )
        parsed = parse_solution_comparison_response(generation)
        attempt = {
            "attempt_idx": attempt_idx,
            "request_seed": req_seed,
            "is_valid": parsed["is_valid"],
            "equivalence_score": parsed["equivalence_score"],
            "equivalence_score_raw": parsed["equivalence_score_raw"],
            "analysis": parsed["analysis"],
        }
        if keep_attempt_generations:
            attempt["generation"] = parsed["clean_generation"]
        attempts.append(attempt)
        if parsed["is_valid"]:
            return {
                "status": "success",
                "selected_attempt_idx": attempt_idx,
                "equivalence_score": parsed["equivalence_score"],
                "analysis": parsed["analysis"],
                "attempts": attempts,
            }

    return {
        "status": "failed",
        "selected_attempt_idx": None,
        "equivalence_score": None,
        "analysis": None,
        "attempts": attempts,
    }
