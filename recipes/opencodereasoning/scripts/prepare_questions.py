import argparse
import copy
import json
import os
import gc
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

def get_question_and_solution(ds_name, ds_split, ds_index, loaded_datasets, use_dataset_solution=False):
    if ds_name not in loaded_datasets:
        print(f"Loading dataset: {ds_name}")
        if ds_name == "taco":
            loaded_datasets[ds_name] = load_dataset("BAAI/TACO", trust_remote_code=True)
        elif ds_name == "apps":
            loaded_datasets[ds_name] = load_dataset("codeparrot/apps", trust_remote_code=True)
        elif ds_name == "code_contests":
            loaded_datasets[ds_name] = load_dataset("deepmind/code_contests")
        elif ds_name == "open-r1/codeforces":
            loaded_datasets[ds_name] = load_dataset("open-r1/codeforces")
        else:
            return None, None

    benchmark = loaded_datasets[ds_name][ds_split][int(ds_index)]
    question = None
    solution = ""

    if ds_name == "code_contests":
        question = benchmark.get("description")
        if use_dataset_solution and benchmark.get("solutions"):
            # Try to find a C++ solution if available
            cpp_solutions = [s["solution"] for s in benchmark["solutions"] if s.get("language") == 2] # 2 is often C++ in CC
            if cpp_solutions:
                solution = cpp_solutions[0]
            elif benchmark["solutions"]:
                solution = benchmark["solutions"][0]["solution"]
    elif ds_name == "taco":
        question = benchmark.get("question")
        if use_dataset_solution and benchmark.get("solutions"):
            try:
                sols = json.loads(benchmark["solutions"])
                if sols: solution = sols[0]
            except: pass
    elif ds_name == "apps":
        question = benchmark.get("question")
        if use_dataset_solution and benchmark.get("solutions"):
            try:
                sols = json.loads(benchmark["solutions"])
                if sols: solution = sols[0]
            except: pass
    elif ds_name == "open-r1/codeforces":
        question = benchmark.get("description", "")
        if benchmark.get("input_format"):
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark.get("output_format"):
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark.get("examples"):
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark.get("note"):
            question += "\n\nNote\n\n" + benchmark["note"]
        
        if use_dataset_solution and benchmark.get("solutions"):
            solution = benchmark["solutions"][0]

    return question, solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Open Code Reasoning questions")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the prepared questions")
    parser.add_argument("--use_dataset_solution", action="store_true", help="Use solutions from the original datasets")
    parser.add_argument("--cache_dir", type=str, default=None, help="Custom cache dir to delete after each dataset")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use default HF cache if not specified
    cache_dir = args.cache_dir or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/datasets")

    ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")
    unique_values = set()
    first_occurrence_indices = []

    for split_name in ["cpp"]:
        if split_name not in ocr2_dataset:
            continue
            
        ocr2_ds = ocr2_dataset[split_name]
        items = sorted(list(ocr2_ds), key=lambda x: x["dataset"])
        
        current_dataset_name = None
        loaded_datasets = {}

        for ocr2_ds_item in tqdm(items):
            ds_name = ocr2_ds_item["dataset"]
            
            if ds_name != current_dataset_name:
                if current_dataset_name is not None:
                    print(f"Clearing and deleting disk cache for: {current_dataset_name}")
                    del loaded_datasets[current_dataset_name]
                    gc.collect()
                    # Delete the specific dataset folder from cache to save space
                    # Note: This is aggressive and might require re-downloading if interrupted
                    for root, dirs, files in os.walk(cache_dir):
                        for d in dirs:
                            if ds_name.replace("/", "___") in d or ds_name.split("/")[-1] in d:
                                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

                current_dataset_name = ds_name

            question, solution = get_question_and_solution(
                ds_name, ocr2_ds_item["split"], ocr2_ds_item["index"], 
                loaded_datasets, use_dataset_solution=args.use_dataset_solution
            )
            
            if question:
                ocr2_ds_item["question"] = question
                if args.use_dataset_solution:
                    ocr2_ds_item["solution"] = solution
                else:
                    ocr2_ds_item["solution"] = ""

                ocr2_ds_item["r1_generation"] = ""
                ocr2_ds_item["qwq_critique"] = ""

                if ocr2_ds_item["question_id"] not in unique_values:
                    unique_values.add(ocr2_ds_item["question_id"])
                    first_occurrence_indices.append(copy.deepcopy(ocr2_ds_item))
        
        loaded_datasets.clear()
        gc.collect()

    output_filepath = os.path.join(output_dir, "open_code_reasoning_questions.jsonl")
    with open(output_filepath, "w") as f:
        for item in first_occurrence_indices:
            f.write(json.dumps(item) + "\n")

    print(f"Prepared questions saved to {output_filepath}")
