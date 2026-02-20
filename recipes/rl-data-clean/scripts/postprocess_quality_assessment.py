#!/usr/bin/env python3
"""
Postprocess quality assessment outputs.
Parses model generations and filters based on ACCEPT/REJECT decisions.
"""

import argparse
import re
from typing import Any, Dict, Optional

import jsonlines


def parse_field(text: str, field_name: str) -> Optional[str]:
    """Extract a field value from model output"""
    if not text or not isinstance(text, str):
        return None
    pattern = rf"{re.escape(field_name)}:\s*(.+?)(?=\n[A-Z_][A-Z_\s]*:|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# Valid difficulty values from solve-and-difficulty-combined prompt (order: longer first for TOO_SIMPLE/UNCLEAR)
DIFFICULTY_VALUES = ("TOO_SIMPLE", "UNCLEAR", "P1", "P2", "P3", "P4", "P5", "P6")


def parse_difficulty(generation: str) -> Optional[str]:
    """Extract difficulty from generation. Only returns a single token (P1-P6, TOO_SIMPLE, UNCLEAR)."""
    if not generation or not isinstance(generation, str):
        return None
    # Prefer patterns that capture only the first token after the label (avoid capturing trailing paragraphs)
    for pat in (
        r"DIFFICULTY_ESTIMATE\s*:\s*\*?\*?\s*(\w+)",
        r"DIFFICULTY\s+ESTIMATE\s*\*?\*?\s*:\s*(\w+)",
        r"\*\*DIFFICULTY\s+ESTIMATE\*\*:\s*(\w+)",
        r"DIFFICULTY\s+ESTIMATE\*\*:\s*(\w+)",
    ):
        m = re.search(pat, generation, re.IGNORECASE)
        if m:
            v = _normalize_difficulty(m.group(1))
            if v:
                return v
    # Fallback: take first line after label and extract first valid token (handles "** P5  \n...")
    for label in ("DIFFICULTY_ESTIMATE", "DIFFICULTY ESTIMATE"):
        pattern = rf"{re.escape(label)}\s*:\s*\*?\*?\s*([^\n]+)"
        m = re.search(pattern, generation, re.IGNORECASE)
        if m:
            first_line = m.group(1).strip()
            v = _normalize_difficulty(first_line)
            if v:
                return v
    return None


def _normalize_difficulty(s: str) -> Optional[str]:
    """Return exactly one of P1-P6, TOO_SIMPLE, UNCLEAR; never return a paragraph."""
    if not s:
        return None
    s = s.strip().upper()
    # Exact match
    if s in DIFFICULTY_VALUES:
        return s
    # First token that is a valid difficulty (e.g. "** P5  \\n..." -> take "P5")
    for token in re.split(r"[\s*]+", s):
        if token in DIFFICULTY_VALUES:
            return token
    return None


def parse_decision(generation: str) -> str:
    """Parse DECISION field from model output"""
    decision = parse_field(generation, "DECISION")
    if decision:
        decision_clean = decision.upper().strip()
        if "ACCEPT" in decision_clean:
            return "ACCEPT"
        elif "REJECT" in decision_clean:
            return "REJECT"
    return "UNKNOWN"


def parse_problem_quality(generation: str) -> Dict[str, Any]:
    """Parse problem quality assessment output (solve-and-difficulty-combined)."""
    return {
        "decision": parse_decision(generation),
        "clarity_analysis": parse_field(generation, "CLARITY_ANALYSIS"),
        "completeness_analysis": parse_field(generation, "COMPLETENESS_ANALYSIS"),
        "rigor_analysis": parse_field(generation, "MATHEMATICAL_RIGOR_ANALYSIS"),
        "difficulty": parse_difficulty(generation),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def parse_discussion_quality(generation: str) -> Dict[str, Any]:
    """Parse discussion quality assessment output"""
    return {
        "decision": parse_decision(generation),
        "meaningful_content_analysis": parse_field(generation, "MEANINGFUL_CONTENT_ANALYSIS"),
        "solution_presence_analysis": parse_field(generation, "SOLUTION_PRESENCE_ANALYSIS"),
        "solution_clarity_analysis": parse_field(generation, "SOLUTION_CLARITY_ANALYSIS"),
        "coherence_analysis": parse_field(generation, "DISCUSSION_COHERENCE_ANALYSIS"),
        "multiple_approaches": parse_field(generation, "MULTIPLE_APPROACHES"),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def parse_proof_quality(generation: str) -> Dict[str, Any]:
    """Parse proof quality assessment output"""
    return {
        "decision": parse_decision(generation),
        "correctness_analysis": parse_field(generation, "CORRECTNESS_ANALYSIS"),
        "rigor_completeness_analysis": parse_field(generation, "RIGOR_AND_COMPLETENESS_ANALYSIS"),
        "clarity_analysis": parse_field(generation, "CLARITY_ANALYSIS"),
        "insight_analysis": parse_field(generation, "MATHEMATICAL_INSIGHT_ANALYSIS"),
        "multiple_approaches": parse_field(generation, "MULTIPLE_APPROACHES"),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def parse_imo_readiness(generation: str) -> Dict[str, Any]:
    """Parse IMO readiness assessment output"""
    return {
        "decision": parse_decision(generation),
        "olympiad_style_analysis": parse_field(generation, "OLYMPIAD_STYLE_ANALYSIS"),
        "pedagogical_value_analysis": parse_field(generation, "PEDAGOGICAL_VALUE_ANALYSIS"),
        "difficulty_analysis": parse_field(generation, "DIFFICULTY_APPROPRIATENESS_ANALYSIS"),
        "teachability_analysis": parse_field(generation, "PROOF_TEACHABILITY_ANALYSIS"),
        "training_signal_analysis": parse_field(generation, "TRAINING_SIGNAL_QUALITY_ANALYSIS"),
        "overall_synthesis": parse_field(generation, "OVERALL_QUALITY_SYNTHESIS"),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def parse_complete_solution_quality(generation: str) -> Dict[str, Any]:
    """Parse complete solution quality assessment output (non-proof problems)"""
    return {
        "decision": parse_decision(generation),
        "problem_quality_analysis": parse_field(generation, "PROBLEM_QUALITY_ANALYSIS"),
        "solution_presence_clarity_analysis": parse_field(generation, "SOLUTION_PRESENCE_CLARITY_ANALYSIS"),
        "solution_correctness_analysis": parse_field(generation, "SOLUTION_CORRECTNESS_ANALYSIS"),
        "solution_completeness_analysis": parse_field(generation, "SOLUTION_COMPLETENESS_ANALYSIS"),
        "consistency_analysis": parse_field(generation, "CONSISTENCY_ANALYSIS"),
        "rl_training_suitability_analysis": parse_field(generation, "RL_TRAINING_SUITABILITY_ANALYSIS"),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def parse_problem_only_quality(generation: str) -> Dict[str, Any]:
    """Parse problem-only quality assessment output"""
    return {
        "decision": parse_decision(generation),
        "clarity_and_completeness_analysis": parse_field(generation, "CLARITY_AND_COMPLETENESS_ANALYSIS"),
        "mathematical_correctness_analysis": parse_field(generation, "MATHEMATICAL_CORRECTNESS_ANALYSIS"),
        "problem_type_and_difficulty": parse_field(generation, "PROBLEM_TYPE_AND_DIFFICULTY"),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def parse_problem_answer_quality(generation: str) -> Dict[str, Any]:
    """Parse problem-answer quality assessment output (quick screening)"""
    return {
        "decision": parse_decision(generation),
        "problem_quality_analysis": parse_field(generation, "PROBLEM_QUALITY_ANALYSIS"),
        "answer_quality_analysis": parse_field(generation, "ANSWER_QUALITY_ANALYSIS"),
        "consistency_analysis": parse_field(generation, "CONSISTENCY_ANALYSIS"),
        "critical_issues": parse_field(generation, "CRITICAL_ISSUES"),
        "decision_reasoning": parse_field(generation, "DECISION_REASONING"),
    }


def filter_by_decision(
    input_file: str,
    output_accept: str,
    output_reject: str,
    stage: str,
):
    """Filter items based on ACCEPT/REJECT decisions"""

    parse_funcs = {
        "problem_quality": parse_problem_quality,
        "discussion_quality": parse_discussion_quality,
        "proof_quality": parse_proof_quality,
        "imo_readiness": parse_imo_readiness,
        "problem_only_quality": parse_problem_only_quality,
        "problem_answer_quality": parse_problem_answer_quality,
        "complete_solution_quality": parse_complete_solution_quality,
    }

    parse_func = parse_funcs[stage]

    accepted = []
    rejected = []
    unknown = []

    with jsonlines.open(input_file) as reader:
        for item in reader:
            # Support both old format (generation) and new format (serialized_output)
            generation = item.get("generation", "")
            if not generation and "serialized_output" in item:
                # Try to get from serialized_output[0]['content']
                serialized = item.get("serialized_output", [])
                if serialized and len(serialized) > 0:
                    content = serialized[0].get("content", "")
                    # Ensure we get a string, not None
                    generation = content if content else ""

            # Also try reasoning_content as last resort
            if not generation and "reasoning_content" in item:
                reasoning = item.get("reasoning_content", "")
                generation = reasoning if reasoning else ""

            # Skip items with no generation content
            if not generation or not isinstance(generation, str):
                print("Warning: Skipping item with no valid generation content")
                continue

            assessment = parse_func(generation)

            # Add assessment to item
            item["quality_assessment"] = assessment

            # Filter by decision
            decision = assessment.get("decision", "UNKNOWN")
            if decision == "ACCEPT":
                accepted.append(item)
            elif decision == "REJECT":
                rejected.append(item)
            else:
                # Treat UNKNOWN as REJECT for safety
                unknown.append(item)
                rejected.append(item)

    # Write outputs
    with jsonlines.open(output_accept, "w") as writer:
        writer.write_all(accepted)

    with jsonlines.open(output_reject, "w") as writer:
        writer.write_all(rejected)

    total = len(accepted) + len(rejected)
    print(f"\n{'=' * 60}")
    print(f"Stage: {stage}")
    print(f"{'=' * 60}")
    print(f"Total items: {total}")
    print(f"Accepted: {len(accepted)} ({len(accepted) / total * 100:.1f}%)")
    print(f"Rejected: {len(rejected)} ({len(rejected) / total * 100:.1f}%)")
    if unknown:
        print(f"Unknown (treated as rejected): {len(unknown)} ({len(unknown) / total * 100:.1f}%)")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Filter items by ACCEPT/REJECT decision")
    parser.add_argument("input_file", help="Input JSONL file with generations")
    parser.add_argument("output_accept", help="Output file for accepted items")
    parser.add_argument("output_reject", help="Output file for rejected items")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "problem_quality",
            "discussion_quality",
            "proof_quality",
            "imo_readiness",
            "problem_only_quality",
            "problem_answer_quality",
            "complete_solution_quality",
        ],
        help="Which assessment stage",
    )

    args = parser.parse_args()

    filter_by_decision(
        args.input_file,
        args.output_accept,
        args.output_reject,
        args.stage,
    )


if __name__ == "__main__":
    main()
