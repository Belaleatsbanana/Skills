import json
from pathlib import Path

from nemo_skills.pipeline.cli import eval, generate, run_cmd, wrap_arguments

def eval_gptoss_tool():
    max_tokens = 120000
    cluster = "ord"
    n_rs = 64
    benchmarks = f"apex-shortlist:{n_rs}"
    model = "/hf_models/gpt-oss-120b"
    n_run = 1
    expname = f"gpt-oss-eval-tool-{n_run}"
    output_dir = "/workspace/tt-compute/evals"
    eval(
            ctx=wrap_arguments(
                f"++inference.tokens_to_generate={max_tokens} "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++prompt_config=gpt-oss/math "
                "++inference.endpoint_type=text "
                "++code_tags=gpt-oss "
                "++code_execution=true "
                "++server.code_execution.max_code_executions=100 "
                "++server.code_execution.code_execution_timeout=120.0 "
                "++chat_template_kwargs.reasoning_effort=high "
                "++chat_template_kwargs.builtin_tools=[python] "
            ),
            cluster=cluster,
            benchmarks=benchmarks,
            model=model,
            server_gpus=8,
            num_jobs=1,
            server_type="vllm",
            output_dir=output_dir,
            server_args="--async-scheduling",
            with_sandbox=True,
            expname=expname,
            wandb_project=expname,
            wandb_name=expname,
            partition="interactive",
        )
    
def eval_gptoss_notool(benchmark, n_rs):
    max_tokens = 120000
    cluster = "iad"
    benchmarks = f"{benchmark}:{n_rs}"
    model = "/hf_models/gpt-oss-120b"
    n_run = 1
    expname = f"gpt-oss-eval-notool-{benchmark}-{n_rs}-run{n_run}"
    output_dir = "/workspace/tt-compute/evals"
    inference_params = (
        f"++inference.tokens_to_generate={max_tokens} "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++inference.endpoint_type=text "
        "++chat_template_kwargs.reasoning_effort=high "
    )
    eval(
            ctx=wrap_arguments(
                f"{inference_params} "
                "++prompt_config=gpt-oss/math "
                # "++code_tags=gpt-oss "
                # "++code_execution=true "
                # "++server.code_execution.max_code_executions=100 "
                # "++server.code_execution.code_execution_timeout=120.0 "
                # "++chat_template_kwargs.builtin_tools=[python] "
            ),
            cluster=cluster,
            benchmarks=benchmarks,
            model=model,
            server_gpus=8,
            # num_jobs=1,
            server_type="vllm",
            output_dir=output_dir,
            server_args="--async-scheduling",
            # judge_server_address="https://inference-api.nvidia.com/v1/chat/completions",
            # judge_model=model,
            # judge_server_type="vllm",
            # judge_server_gpus=8,
            # judge_generation_type="math_judge",
            # extra_judge_args=(
            #     f"{inference_params} "
            #     "++inference.reasoning_effort=null "
            # ),
            starting_seed=440,
            # with_sandbox=True,
            expname=expname,
            wandb_project=expname,
            wandb_name=expname,
            # partition="interactive",
        )


def _score_to_expert_rating(sample: dict) -> float:
    """Map per-sample correctness to 0-7 expert rating scale for BON metrics."""
    if "symbolic_correct" in sample:
        return 7.0 if bool(sample["symbolic_correct"]) else 0.0
    if "judge_correct" in sample:
        return 7.0 if bool(sample["judge_correct"]) else 0.0
    return 0.0


def _build_bon_input_from_rs_outputs(
    benchmark: str,
    input_dir: Path,
    output_file: Path,
    n_rs: int = 64,
    force_rebuild: bool = False,
):
    """
    Convert output-rs*.jsonl files (one candidate per seed) into BON format:
    one record per problem with `proofs` and `expert_ratings` lists.
    """
    if output_file.exists() and not force_rebuild:
        print(f"[{benchmark}] BON input already exists, skipping conversion: {output_file}")
        return

    all_seed_rows = []
    for rs in range(n_rs):
        rs_file = input_dir / f"output-rs{rs}.jsonl"
        if not rs_file.exists():
            raise FileNotFoundError(f"Missing seed file: {rs_file}")
        with rs_file.open("rt", encoding="utf-8") as f:
            all_seed_rows.append([json.loads(line) for line in f])

    if not all_seed_rows:
        raise ValueError(f"No seed files found in {input_dir}")

    num_rows = len(all_seed_rows[0])
    if any(len(rows) != num_rows for rows in all_seed_rows):
        raise ValueError(f"Inconsistent row counts in {input_dir}")

    output_rows = []
    for i in range(num_rows):
        row0 = all_seed_rows[0][i]
        proofs = []
        expert_ratings = []
        for rs_rows in all_seed_rows:
            sample = rs_rows[i]
            proofs.append(sample.get("summary") or _extract_summary_from_generation(sample.get("generation", "")))
            expert_ratings.append(_score_to_expert_rating(sample))

        problem_id = row0.get("problem_id") or row0.get("id") or f"{benchmark}-{i}"
        output_row = {
            "problem_id": str(problem_id),
            "problem": row0["problem"],
            "expected_answer": row0.get("expected_answer"),
            "proofs": proofs,
            "expert_ratings": expert_ratings,
        }
        # Keep optional context fields if present.
        for k in ("source", "category", "subcategory"):
            if k in row0:
                output_row[k] = row0[k]
        output_rows.append(output_row)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wt", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    print(f"[{benchmark}] wrote BON input: {output_file} ({len(output_rows)} problems, {n_rs} candidates each)")


def _extract_summary_from_generation(generation: str) -> str:
    """
    Extract the solution summary from a full generation.
    For gpt-oss-120b, the summary is the assistant final message block.
    """
    if not generation:
        return ""
    marker = "<|start|>assistant<|channel|>final<|message|>"
    if marker in generation:
        summary = generation.rsplit(marker, 1)[1]
    else:
        summary = ""
    summary = summary.split("<|end|>", 1)[0]
    return summary.strip()


def _get_gptoss_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required to count gpt-oss tokens") from exc
    return AutoTokenizer.from_pretrained(
        "/hf_models/gpt-oss-120b",
        local_files_only=True,
        trust_remote_code=True,
    )


def _count_tokens_with_tokenizer(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def add_summary_fields_to_rs_outputs(
    benchmark: str,
    input_dir: Path,
    n_rs: int = 64,
    force_rebuild: bool = False,
):
    """
    Read output-rs*.jsonl, extract solution summary from generation, and
    update each file in-place with fields: summary, num_summary_tokens.
    """
    tokenizer = _get_gptoss_tokenizer()

    for rs in range(n_rs):
        in_file = input_dir / f"output-rs{rs}.jsonl"
        if not in_file.exists():
            raise FileNotFoundError(f"Missing seed file: {in_file}")
        temp_file = input_dir / f"output-rs{rs}.jsonl.tmp"

        if in_file.exists() and not force_rebuild:
            # Skip if file already has summary fields on all rows.
            with in_file.open("rt", encoding="utf-8") as f_in:
                has_missing_summary = False
                for line in f_in:
                    row = json.loads(line)
                    if "summary" not in row or "num_summary_tokens" not in row:
                        has_missing_summary = True
                        break
            if not has_missing_summary:
                print(f"[{benchmark}] summary fields already present, skipping: {in_file}")
                continue

        with in_file.open("rt", encoding="utf-8") as f_in, temp_file.open("wt", encoding="utf-8") as f_out:
            for line in f_in:
                row = json.loads(line)
                generation = row.get("generation", "")
                summary = _extract_summary_from_generation(generation)
                row["summary"] = summary
                row["num_summary_tokens"] = _count_tokens_with_tokenizer(tokenizer, summary)
                f_out.write(json.dumps(row) + "\n")

        temp_file.replace(in_file)
        print(f"[{benchmark}] updated in-place with summary fields: {in_file}")


def prepare_summary_fields_from_existing_candidates(
    benchmarks: list[str] | None = None,
    n_candidates: int = 64,
    force_rebuild: bool = False,
):
    """
    Extract summary fields from output-rs*.jsonl and update files in-place.
    Run separately (ideally on iad) before scheduling eval jobs.
    """
    src_root = Path("/workspace/tt-compute/evals/eval-results")
    benchmarks = benchmarks or ["apex-shortlist"]

    for benchmark in benchmarks:
        rs_dir = src_root / benchmark
        add_summary_fields_to_rs_outputs(
            benchmark=benchmark,
            input_dir=rs_dir,
            n_rs=n_candidates,
            force_rebuild=force_rebuild,
        )


def submit_prepare_summaries_on_iad(
    benchmarks: list[str] | None = None,
    n_candidates: int = 64,
    force_rebuild: bool = False,
    expname: str = "prepare-summaries",
):
    """
    Submit summary extraction as a remote run_cmd job on iad.
    """
    benchmarks = benchmarks or ["apex-shortlist"]
    cmd = (
        "python -c \""
        "import importlib.util; "
        "spec=importlib.util.spec_from_file_location('eval_tt', "
        "'/nemo_run/code/recipes/proof-gen-verification/pipeline/eval_tt.py'); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"m.prepare_summary_fields_from_existing_candidates(benchmarks={benchmarks!r}, "
        f"n_candidates={n_candidates}, force_rebuild={force_rebuild})"
        "\""
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster="iad",
        expname=expname,
    )

def prepare_bon_inputs_from_existing_candidates(
    benchmarks: list[str] | None = None,
    n_candidates: int = 64,
    force_rebuild: bool = False,
):
    """
    Step 1: Build BON inputs from existing output-rs*.jsonl.
    Run this separately (ideally on iad) before scheduling eval jobs.
    """
    src_root = Path("/workspace/tt-compute/evals/eval-results")
    dst_root = Path("/workspace/tt-compute/generic-bon-eval")
    benchmarks = benchmarks or ["apex-shortlist", "imo-answerbench"]

    for benchmark in benchmarks:
        rs_dir = src_root / benchmark
        bon_input_file = dst_root / "inputs" / f"{benchmark}-gpt-oss-120b-rs{n_candidates}.jsonl"
        _build_bon_input_from_rs_outputs(
            benchmark=benchmark,
            input_dir=rs_dir,
            output_file=bon_input_file,
            n_rs=n_candidates,
            force_rebuild=force_rebuild,
        )


def submit_prepare_bon_inputs_on_iad(
    benchmarks: list[str] | None = None,
    n_candidates: int = 64,
    force_rebuild: bool = False,
    expname: str = "prepare-bon-inputs",
):
    """
    Submit BON input preparation as a remote run_cmd job on iad.
    Useful when local machine does not have direct data access.
    """
    benchmarks = benchmarks or ["apex-shortlist", "imo-answerbench"]
    # Import by file path to avoid issues with '-' in directory name.
    cmd = (
        "python -c \""
        "import importlib.util; "
        "spec=importlib.util.spec_from_file_location('eval_tt', "
        "'/nemo_run/code/recipes/proof-gen-verification/pipeline/eval_tt.py'); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        f"m.prepare_bon_inputs_from_existing_candidates(benchmarks={benchmarks!r}, "
        f"n_candidates={n_candidates}, force_rebuild={force_rebuild})"
        "\""
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster="iad",
        expname=expname,
    )


def run_generic_bon_from_prepared_inputs():
    """
    Step 2: Run BON-style genselect eval on iad using prebuilt BON inputs.
    Outputs: /workspace/tt-compute/generic-bon-eval
    """
    dst_root = Path("/workspace/tt-compute/generic-bon-eval")

    cluster = "iad"
    model = "/hf_models/gpt-oss-120b"
    n_candidates = 64
    num_eval_seeds = 8
    num_chunks = 2
    num_shuffles = 500

    # Match eval_judge.py gpt-oss config.
    server_type = "sglang"
    server_gpus = 8
    server_nodes = 1
    server_args = "--ep-size 8 --context-length 128000 --reasoning-parser gpt-oss"
    inline_args = (
        "++inference.tokens_to_generate=8000 "
        "++inference.reasoning_effort=high "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
    )

    eval_types = ["genselect"]

    # Same prompt paths used in configs/judge-eval.yaml.
    prompt_paths = {
        "judgement_binary_prompt_config_path": "/nemo_run/code/recipes/proof-gen-verification/prompts/math_judge/opc_judge_summary.yaml",
        "judgement_binary_prompt_config_path_v2": "/nemo_run/code/recipes/proof-gen-verification/prompts/math_judge/proofbench_none_binary.yaml",
        "judgement_binary_gt_proof_prompt_config_path": "/nemo_run/code/recipes/proof-gen-verification/prompts/math_judge/opc_judge_summary_gt_proof.yaml",
        "judgement_scoring_prompt_config_path": "/nemo_run/code/recipes/proof-gen-verification/prompts/math_judge/proofbench_none.yaml",
        "judgement_scoring_rubric_gt_proof_prompt_config_path": "/nemo_run/code/recipes/proof-gen-verification/prompts/math_judge/proofbench_ms_ref.yaml",
        "genselect_prompt_config_path": "/nemo_run/code/recipes/proof-gen-verification/prompts/genselect/proof_genselect_default.yaml",
    }

    benchmarks = ["apex-shortlist", "imo-answerbench"]

    for benchmark in benchmarks:
        bon_input_file = dst_root / "inputs" / f"{benchmark}-gpt-oss-120b-rs{n_candidates}.jsonl"

        for eval_type in eval_types:
            eval_out_dir = dst_root / benchmark / f"gpt-oss-120b_{eval_type}"
            gen_expname = f"generic-bon-{benchmark}-{eval_type}-gen"
            metrics_expname = f"generic-bon-{benchmark}-{eval_type}-metrics"

            generate(
                ctx=wrap_arguments(
                    f"++model_name={model} "
                    f"{inline_args} "
                    "++script_program_path=/nemo_run/code/recipes/proof-gen-verification/scripts/generate_generic_bon_generation.py "
                    f"++script_config.eval_type={eval_type} "
                    "++script_config.judgement_num_seeds=32 "
                    f"++script_config.judgement_binary_prompt_config_path={prompt_paths['judgement_binary_prompt_config_path']} "
                    f"++script_config.judgement_binary_prompt_config_path_v2={prompt_paths['judgement_binary_prompt_config_path_v2']} "
                    f"++script_config.judgement_binary_gt_proof_prompt_config_path={prompt_paths['judgement_binary_gt_proof_prompt_config_path']} "
                    f"++script_config.judgement_scoring_prompt_config_path={prompt_paths['judgement_scoring_prompt_config_path']} "
                    "++script_config.judgement_scoring_rubric_gt_proof_prompt_config_path="
                    f"{prompt_paths['judgement_scoring_rubric_gt_proof_prompt_config_path']} "
                    f"++script_config.genselect_prompt_config_path={prompt_paths['genselect_prompt_config_path']} "
                    "++max_concurrent_requests=150 "
                    "++enable_litellm_cache=True "
                ),
                generation_module="recipes/proof-gen-verification/scripts/script_generation.py",
                model=model,
                cluster=cluster,
                input_file=str(bon_input_file),
                output_dir=str(eval_out_dir),
                num_chunks=num_chunks,
                num_random_seeds=num_eval_seeds,
                expname=gen_expname,
                server_type=server_type,
                server_gpus=server_gpus,
                server_nodes=server_nodes,
                server_args=server_args,
                exclusive=True,
            )

            run_cmd(
                ctx=wrap_arguments(
                    "python /nemo_run/code/recipes/proof-gen-verification/scripts/generic_eval_bon.py "
                    f"--input_dir {eval_out_dir} "
                    f"--output_file {eval_out_dir}/bon_metrics.json "
                    f"--num_shuffles {num_shuffles} "
                ),
                cluster=cluster,
                expname=metrics_expname,
                run_after=[gen_expname],
            )

def genparallel_offline(mode, benchmark, n_sol):
    cluster = "iad"
    generation_dir = f"/workspace/tt-compute/evals/eval-results/{benchmark}"
    output_dir = f"/workspace/tt-compute/parallel/{mode}"
    max_tokens = 120000
    model = "/hf_models/gpt-oss-120b"
    n_run = 1
    expname = f"gpt-oss-parallel-{mode}-{benchmark}-{n_sol}-run{n_run}"
    prompt_path = "/nemo_run/code/nemo_skills/prompt/config/generic/genselect.yaml"
    end_reasoning = "<|start|>assistant<|channel|>final<|message|>"
    inference_params = (
        f"++inference.tokens_to_generate={max_tokens} "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++inference.endpoint_type=text "
        "++chat_template_kwargs.reasoning_effort=high "
    )
    parallel_params = (
        f"++parallel_thinking.mode={mode} "
        f"++parallel_thinking.generation_dir={generation_dir} "
        f"++parallel_thinking.num_initial_solutions={n_sol} "
        f"++parallel_thinking.genselect.prompt_config={prompt_path} "
        "++parallel_thinking.solution_key=summary "
        "++parallel_thinking.parse_reasoning_solutions=False "
        # f"++parallel_thinking.end_reasoning_string='{end_reasoning}' "
        "++parallel_thinking.endpoint_type=text "
    )
    eval(
            ctx=wrap_arguments(
                f"{inference_params} "
                "++prompt_config=gpt-oss/math "
                f"{parallel_params} "
                # "++code_tags=gpt-oss "
                # "++code_execution=true "
                # "++server.code_execution.max_code_executions=100 "
                # "++server.code_execution.code_execution_timeout=120.0 "
                # "++chat_template_kwargs.builtin_tools=[python] "
            ),
            cluster=cluster,
            benchmarks=f"{benchmark}:1",
            model=model,
            server_gpus=8,
            # num_jobs=1,
            server_type="vllm",
            output_dir=output_dir,
            server_args="--async-scheduling",
            # judge_server_address="https://inference-api.nvidia.com/v1/chat/completions",
            # judge_model=model,
            # judge_server_type="vllm",
            # judge_server_gpus=8,
            # judge_generation_type="math_judge",
            # extra_judge_args=(
            #     f"{inference_params} "
            #     "++inference.reasoning_effort=null "
            # ),
            # starting_seed=440,
            # with_sandbox=True,
            expname=expname,
            wandb_project=expname,
            wandb_name=expname,
            partition="interactive",
        )

if __name__ == "__main__":
    # eval_gptoss_tool()
    # eval_gptoss_notool(benchmark="apex-shortlist", n_rs=1)
    # Optional remote submit (runs conversion on iad):
    # submit_prepare_bon_inputs_on_iad(benchmarks=["imo-answerbench"], force_rebuild=False, expname="prepare-bon-inputs-imo-answerbench")
    # Step 1 (run once, preferably on iad):
    # prepare_bon_inputs_from_existing_candidates(benchmarks=["apex-shortlist"], force_rebuild=False)
    # Step 2:
    # run_generic_bon_from_prepared_inputs()
    # submit_prepare_summaries_on_iad(benchmarks=["apex-shortlist"], n_candidates=512, force_rebuild=True, expname="prepare-summaries-answerbench")
    genparallel_offline(mode="genselect", benchmark="apex-shortlist", n_sol=2)
