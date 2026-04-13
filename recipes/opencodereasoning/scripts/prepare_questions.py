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
    """
    Helper to fetch the actual question text from the source datasets (TACO, APPS, etc.)
    using the index mapping provided by OpenCodeReasoning-2.
    """
    if ds_name not in loaded_datasets:
        # --- TRIGGERED ON DATASET CHANGE ---
        # This block runs when we move to a new source dataset (e.g., from 'apps' to 'taco')
        print(f"\n[STEP] Fetching source dataset from HuggingFace: {ds_name}")
        
        if ds_name == "taco":
            loaded_datasets[ds_name] = load_dataset("BAAI/TACO", trust_remote_code=True)
        elif ds_name == "apps":
            loaded_datasets[ds_name] = load_dataset("codeparrot/apps", trust_remote_code=True)
        elif ds_name == "code_contests":
            # NOTE: CodeContests is huge (~100GB+). This is where the most disk space is used.
            loaded_datasets[ds_name] = load_dataset("deepmind/code_contests")
        elif ds_name == "open-r1/codeforces":
            loaded_datasets[ds_name] = load_dataset("open-r1/codeforces")
        else:
            return None, None

    # Access the specific question using the index from OCR2
    benchmark = loaded_datasets[ds_name][ds_split][int(ds_index)]
    question = None
    solution = ""

    # Extracting logic varies per dataset schema
    if ds_name == "code_contests":
        question = benchmark.get("description")
        if use_dataset_solution and benchmark.get("solutions"):
            # Language 2 is C++ in CodeContests
            cpp_solutions = [s["solution"] for s in benchmark["solutions"] if s.get("language") == 2]
            solution = cpp_solutions[0] if cpp_solutions else benchmark["solutions"][0]["solution"]
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
        # Codeforces requires concatenating multiple fields for a full prompt
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

    # Resolve HF cache directory for cleanup
    cache_dir = args.cache_dir or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/datasets")

    # [ACTION] Load the main OCR2 mapping dataset
    print("[STEP 1] Loading OpenCodeReasoning-2 'cpp' split...")
    ocr2_dataset = load_dataset(
    "parquet",
    data_files="hf://datasets/nvidia/OpenCodeReasoning-2/cpp/train-*.parquet",
    split="train"
    )
    unique_values = set()
    first_occurrence_indices = []

    # [ACTION] Extremely fast sorting using Arrow
    # We avoid converting to Pandas or Python lists here. 
    # Arrow (the backend of datasets) can sort the table on disk/memory very efficiently.
    print("[STEP 2] Sorting questions by dataset using Arrow (High Speed)...")
    # This keeps the data in the optimized Arrow format while sorting
    ocr2_dataset = ocr2_dataset.sort("dataset")
    
    current_dataset_name = None
    loaded_datasets = {}

    # We iterate directly over the dataset object to avoid any large memory conversions
    print(f"[STEP 3] Starting extraction of {len(ocr2_dataset)} questions...")
    for ocr2_ds_item in tqdm(ocr2_dataset, desc="Processing Questions"):
        ds_name = ocr2_ds_item["dataset"]
        
        # Check if we have switched to a new source dataset
        if ds_name != current_dataset_name:
            if current_dataset_name is not None:
                # --- MEMORY & DISK CLEANUP ---
                print(f"\n[CLEANUP] Finished {current_dataset_name}. Clearing memory and disk cache...")
                
                # 1. Remove from RAM
                if current_dataset_name in loaded_datasets:
                    del loaded_datasets[current_dataset_name]
                gc.collect()
                
                # 2. Remove from Disk
                # We look for the folder name in the HF cache that matches the dataset name
                for root, dirs, files in os.walk(cache_dir):
                    for d in dirs:
                        # HF names folders like 'BAAI___TACO' or 'codeparrot___apps'
                        search_term = current_dataset_name.replace("/", "___")
                        if search_term in d or current_dataset_name.split("/")[-1] in d:
                            target = os.path.join(root, d)
                            print(f"[DISK] Deleting cache folder: {target}")
                            shutil.rmtree(target, ignore_errors=True)

            current_dataset_name = ds_name

        # [ACTION] Fetch question text from the source dataset
        question, solution = get_question_and_solution(
            ds_name, ocr2_ds_item["split"], ocr2_ds_item["index"], 
            loaded_datasets, use_dataset_solution=args.use_dataset_solution
        )
        
        if question:
            # Add the full question text to the OCR2 metadata object
            ocr2_ds_item["question"] = question
            if args.use_dataset_solution:
                ocr2_ds_item["solution"] = solution
            else:
                ocr2_ds_item["solution"] = ""

            # Placeholders for the next stages in the pipeline
            ocr2_ds_item["r1_generation"] = ""
            ocr2_ds_item["qwq_critique"] = ""

            # Deduplication check
            if ocr2_ds_item["question_id"] not in unique_values:
                unique_values.add(ocr2_ds_item["question_id"])
                first_occurrence_indices.append(copy.deepcopy(ocr2_ds_item))
    
    # Final cleanup for the last dataset processed
    if loaded_datasets:
        loaded_datasets.clear()
        gc.collect()

    # [STEP 4] Save the final compiled dataset
    output_filepath = os.path.join(output_dir, "open_code_reasoning_questions.jsonl")
    print(f"[STEP 4] Saving {len(first_occurrence_indices)} unique questions to {output_filepath}")
    with open(output_filepath, "w") as f:
        for item in first_occurrence_indices:
            f.write(json.dumps(item) + "\n")

    print("\n[SUCCESS] Pipeline step 'Prepare Questions' complete.")
