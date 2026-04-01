import asyncio
import re
from typing import Any

import numpy as np
import omegaconf

from nemo_skills.inference.model import BaseModel


async def llm_call(
    llm: BaseModel,
    prompt: str,
    llm_kwargs: dict[str, Any],
    req_seed: int,
) -> tuple[str, dict[str, Any]]:
    """Run one generation call and return cleaned text plus metadata."""
    response = await llm.generate_async(
        prompt=[{"role": "user", "content": prompt}],
        **llm_kwargs,
        random_seed=req_seed,
    )
    full_response = response["generation"]
    output_text = full_response.split("</think>")[-1].strip()
    return output_text, {"num_generated_tokens": response.get("num_generated_tokens", 0), "seed": req_seed}


def extract_verification_score(verification_text: str) -> float | None:
    """Extract verifier score from text. Supports boxed score 0/0.5/1."""
    if not verification_text:
        return None

    boxed_patterns = [
        r"\\boxed\{\s*(0(?:\.0)?|0\.5|1(?:\.0)?)\s*\}",
        r"\\boxed\{\{\s*(0(?:\.0)?|0\.5|1(?:\.0)?)\s*\}\}",
    ]
    for pattern in boxed_patterns:
        matches = re.findall(pattern, verification_text)
        if matches:
            return float(matches[-1])

    fallback = re.findall(r"(?<![\d.])(0(?:\.0)?|0\.5|1(?:\.0)?)(?![\d.])", verification_text)
    if fallback:
        return float(fallback[-1])
    return None


def compute_average_verification_score(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute average score over analyses with parsed numeric score."""
    valid_scores = [analysis["score"] for analysis in analyses if analysis.get("score") is not None]
    average_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
    return {
        "average_score": average_score,
        "num_valid_scores": len(valid_scores),
        "num_analyses": len(analyses),
    }


def format_for_refinement(
    proof: str,
    verification_analysis: str,
    verification_score: float | None = None,
) -> str:
    """Build a single solution+verification block for refinement prompts."""
    score_str = "None" if verification_score is None else str(verification_score)
    return (
        "### Candidate Solution\n"
        f"{proof}\n\n"
        "### Verification Analysis\n"
        f"{verification_analysis}\n\n"
        "### Verification Score\n"
        f"{score_str}\n"
    )


def sample_verifications(
    analyses: list[dict[str, Any]],
    k: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    """
    Sample k analyses without replacement, biased toward issue-finding analyses.

    Scores in issue_score_values receive higher sampling weight.
    """
    if k <= 0:
        return []
    if k > len(analyses):
        raise ValueError(f"k={k} must be <= number of analyses={len(analyses)}")
    
    issue_score_values = (0.0, 0.5)
    issue_weight = 2.0
    correct_weight = 1.0
    weights = []
    for analysis in analyses:
        score = analysis.get("score")
        if score in issue_score_values:
            weights.append(issue_weight)
        else:
            weights.append(correct_weight)

    weights = np.asarray(weights, dtype=float)
    probs = weights / weights.sum()
    selected_indices = rng.choice(len(analyses), size=k, replace=False, p=probs)
    return [analyses[int(i)] for i in selected_indices]


async def generate_initial_proofs(
    llm: BaseModel,
    problem: str,
    n_candidates: int,
    proof_gen_prompt_path: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    problem_key_for_prompt: str = "question",
) -> list[dict[str, Any]]:
    """Generate N initial proof candidates for one problem."""
    proof_gen_prompt_template = omegaconf.OmegaConf.load(proof_gen_prompt_path).user
    prompt = proof_gen_prompt_template.format(**{problem_key_for_prompt: problem})
    tasks = [llm_call(llm, prompt, llm_kwargs, int(rng.randint(0, 2**32))) for _ in range(n_candidates)]
    outputs = await asyncio.gather(*tasks)

    return [
        {
            "solution_id": idx,
            "proof": proof,
            "proof_generation_aux": aux,
        }
        for idx, (proof, aux) in enumerate(outputs)
    ]


async def run_verifications_for_proof(
    llm: BaseModel,
    problem: str,
    proof: str,
    n_analyses: int,
    verification_prompt_path: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    problem_key_for_prompt: str = "statement",
    proof_key_for_prompt: str = "proof",
) -> dict[str, Any]:
    """Run M verification analyses for one proof and return parsed scores."""
    verification_prompt_template = omegaconf.OmegaConf.load(verification_prompt_path).user
    prompt = verification_prompt_template.format(**{problem_key_for_prompt: problem, proof_key_for_prompt: proof})
    tasks = [llm_call(llm, prompt, llm_kwargs, int(rng.randint(0, 2**32))) for _ in range(n_analyses)]
    outputs = await asyncio.gather(*tasks)

    analyses = []
    for verification_id, (analysis_text, aux) in enumerate(outputs):
        analyses.append(
            {
                "verification_id": verification_id,
                "analysis": analysis_text,
                "score": extract_verification_score(analysis_text),
                "verification_aux": aux,
            }
        )

    score_summary = compute_average_verification_score(analyses)
    return {
        "analyses": analyses,
        **score_summary,
    }


async def refine_single_solution(
    llm: BaseModel,
    problem: str,
    proof: str,
    verification_analysis: str,
    refinement_prompt_path: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    verification_score: float | None = None,
    problem_key_for_prompt: str = "instruction",
    refinements_key_for_prompt: str = "proofs_to_refine",
) -> dict[str, Any]:
    """
    Refine one proof using one verification analysis.
    """
    refinement_prompt_template = omegaconf.OmegaConf.load(refinement_prompt_path).user
    proofs_to_refine = format_for_refinement(
        proof=proof,
        verification_analysis=verification_analysis,
        verification_score=verification_score,
    )
    prompt = refinement_prompt_template.format(
        **{
            problem_key_for_prompt: problem,
            refinements_key_for_prompt: proofs_to_refine,
        }
    )
    refined_proof, refine_aux = await llm_call(llm, prompt, llm_kwargs, int(rng.randint(0, 2**32)))
    return {
        "proof": refined_proof,
        "refinement_aux": refine_aux,
    }


async def refine_single_solution_verifications(
    llm: BaseModel,
    problem: str,
    solution: dict[str, Any],
    analyses: list[dict[str, Any]],
    k: int,
    refinement_prompt_path: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    problem_key_for_prompt: str = "instruction",
    refinements_key_for_prompt: str = "proofs_to_refine",
) -> list[dict[str, Any]]:
    """
    For one solution, sample k analyses (with issue bias) and produce k refined solutions.
    """
    selected_analyses = sample_verifications(analyses=analyses, k=k, rng=rng)

    parent_solution_id = solution.get("solution_id")
    parent_proof = solution["proof"]

    tasks = []
    for analysis in selected_analyses:
        tasks.append(
            refine_single_solution(
                llm=llm,
                problem=problem,
                proof=parent_proof,
                verification_analysis=analysis.get("analysis", ""),
                verification_score=analysis.get("score"),
                refinement_prompt_path=refinement_prompt_path,
                llm_kwargs=llm_kwargs,
                rng=np.random.RandomState(int(rng.randint(0, 2**32))),
                problem_key_for_prompt=problem_key_for_prompt,
                refinements_key_for_prompt=refinements_key_for_prompt,
            )
        )

    refined_outputs = await asyncio.gather(*tasks)
    refined_solutions = []
    for local_refinement_id, (analysis, output) in enumerate(zip(selected_analyses, refined_outputs)):
        refined_solutions.append(
            {
                "solution_id": f"{parent_solution_id}-r{local_refinement_id}",
                "parent_solution_id": parent_solution_id,
                "source_verification_id": analysis.get("verification_id"),
                "source_verification_score": analysis.get("score"),
                "proof": output["proof"],
                "refinement_aux": output["refinement_aux"],
            }
        )
    return refined_solutions


async def refine_solution_pool(
    llm: BaseModel,
    problem: str,
    solutions: list[dict[str, Any]],
    n_top: int,
    k_refinements_per_solution: int,
    n_verifications_per_solution: int,
    verification_prompt_path: str,
    refinement_prompt_path: str,
    llm_kwargs: dict[str, Any],
    rng: np.random.RandomState,
    verification_problem_key_for_prompt: str = "statement",
    verification_proof_key_for_prompt: str = "proof",
    refinement_problem_key_for_prompt: str = "instruction",
    refinement_refinements_key_for_prompt: str = "proofs_to_refine",
) -> dict[str, Any]:
    """
    Run one full refinement round:
    1) ensure original solutions have verification scores,
    2) refine each solution k times (biased sampling over analyses),
    3) verify refined solutions,
    4) return top-n by average verification score from original+refined pool.
    """
    if n_top <= 0:
        raise ValueError("n_top must be positive")

    # Ensure original solutions have ids + verification results.
    original_solutions = []
    verification_tasks = []
    for i, solution in enumerate(solutions):
        solution_copy = dict(solution)
        if "solution_id" not in solution_copy:
            solution_copy["solution_id"] = f"orig-{i}"
        original_solutions.append(solution_copy)

        if "verification_results" not in solution_copy:
            verification_tasks.append(
                run_verifications_for_proof(
                    llm=llm,
                    problem=problem,
                    proof=solution_copy["proof"],
                    n_analyses=n_verifications_per_solution,
                    verification_prompt_path=verification_prompt_path,
                    llm_kwargs=llm_kwargs,
                    rng=np.random.RandomState(int(rng.randint(0, 2**32))),
                    problem_key_for_prompt=verification_problem_key_for_prompt,
                    proof_key_for_prompt=verification_proof_key_for_prompt,
                )
            )
        else:
            verification_tasks.append(None)

    pending_tasks = [task for task in verification_tasks if task is not None]
    pending_results = await asyncio.gather(*pending_tasks) if pending_tasks else []

    pending_idx = 0
    for i, task in enumerate(verification_tasks):
        if task is not None:
            original_solutions[i]["verification_results"] = pending_results[pending_idx]
            pending_idx += 1

    for solution in original_solutions:
        solution["average_verification_score"] = solution["verification_results"]["average_score"]

    # Refine each original solution using its analyses.
    per_solution_refine_tasks = []
    for solution in original_solutions:
        analyses = solution["verification_results"]["analyses"]
        per_solution_refine_tasks.append(
            refine_single_solution_verifications(
                llm=llm,
                problem=problem,
                solution=solution,
                analyses=analyses,
                k=k_refinements_per_solution,
                refinement_prompt_path=refinement_prompt_path,
                llm_kwargs=llm_kwargs,
                rng=np.random.RandomState(int(rng.randint(0, 2**32))),
                problem_key_for_prompt=refinement_problem_key_for_prompt,
                refinements_key_for_prompt=refinement_refinements_key_for_prompt,
            )
        )

    refined_nested = await asyncio.gather(*per_solution_refine_tasks) if per_solution_refine_tasks else []
    refined_solutions = [item for sublist in refined_nested for item in sublist]

    # Verify refined solutions.
    refined_verification_tasks = [
        run_verifications_for_proof(
            llm=llm,
            problem=problem,
            proof=solution["proof"],
            n_analyses=n_verifications_per_solution,
            verification_prompt_path=verification_prompt_path,
            llm_kwargs=llm_kwargs,
            rng=np.random.RandomState(int(rng.randint(0, 2**32))),
            problem_key_for_prompt=verification_problem_key_for_prompt,
            proof_key_for_prompt=verification_proof_key_for_prompt,
        )
        for solution in refined_solutions
    ]
    refined_verifications = await asyncio.gather(*refined_verification_tasks) if refined_verification_tasks else []

    for solution, verification_results in zip(refined_solutions, refined_verifications):
        solution["verification_results"] = verification_results
        solution["average_verification_score"] = verification_results["average_score"]

    combined_pool = original_solutions + refined_solutions
    combined_pool_sorted = sorted(
        combined_pool,
        key=lambda s: (
            s.get("average_verification_score", 0.0),
            s.get("verification_results", {}).get("num_valid_scores", 0),
        ),
        reverse=True,
    )
    top_solutions = combined_pool_sorted[:n_top]

    return {
        "top_solutions": top_solutions,
        "original_solutions": original_solutions,
        "refined_solutions": refined_solutions,
        "combined_pool_size": len(combined_pool),
    }
