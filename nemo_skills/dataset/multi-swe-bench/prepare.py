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

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from nemo_skills.dataset.utils import download_with_retries

DATASET_NAME = {
    "flash": "ByteDance-Seed/Multi-SWE-bench-flash",
    "mini": "ByteDance-Seed/Multi-SWE-bench_mini",
    "full": "ByteDance-Seed/Multi-SWE-bench",
}


def convert_language(language):
    if language == "c++":
        return "cpp"
    elif language == "js":
        return "javascript"
    elif language == "ts":
        return "typescript"
    else:
        return language


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_image}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time. "
        "See nemo_skills/dataset/swe-bench/dump_images.py",
    )
    parser.add_argument(
        "--subset", type=str, default="full", help="Multi-SWE-bench subset to use. Must be one of: flash, mini, full."
    )
    parser.add_argument(
        "--setup", type=str, default="default", help="Setup name (used as nemo-skills split parameter)."
    )
    args = parser.parse_args()

    subset = args.subset
    container_formatter = args.container_formatter

    # Multi-SWE-bench has weird formatting in their jsonl files, so HF load_dataset doesn't work with it.
    # Instead, we download the jsonl files directly and format them to work with our swebench generation module.

    temp_dir = Path(__file__).parent / ".temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    if subset == "flash":
        data_files = [temp_dir / "flash.jsonl"]
        download_with_retries(
            url="https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench-flash/resolve/main/multi_swe_bench_flash.jsonl",
            output_file=data_files[0],
        )
    elif subset == "mini":
        data_files = [temp_dir / "mini.jsonl"]
        download_with_retries(
            url="https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_mini/resolve/main/multi_swe_bench_mini.jsonl",
            output_file=data_files[0],
        )
    elif subset == "full":
        # Check if git-lfs is installed
        try:
            subprocess.run(
                ["git", "lfs", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("Git LFS is not installed. Please install it to prepare the full Multi-SWE-bench dataset.")
            exit(1)

        # Clone the Multi-SWE-bench dataset
        subprocess.run(
            ["git", "clone", "https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench", str(temp_dir)],
            check=True,
        )

        # Get a list of all jsonl files
        data_files = sorted(list(temp_dir.glob("*/*.jsonl")))
    else:
        raise ValueError(f"Subset '{subset}' is not supported. Must be one of: flash, mini, full.")

    output_file = Path(__file__).parent / f"{args.setup}.jsonl"

    with open(output_file, "w") as fout:
        for data_file in tqdm(data_files, desc="Preparing dataset"):
            with open(data_file, "r") as fin:
                for i, line in enumerate(fin):
                    input_row = json.loads(line)

                    # Multi-SWE-bench docker image format: mswebench/facebook_m_zstd:pr-3942
                    docker_image = f"mswebench/{input_row['org']}_m_{input_row['repo']}:pr-{input_row['number']}"
                    docker_image = docker_image.lower()
                    if container_formatter.endswith(".sif"):
                        docker_image = docker_image.replace("/", "_").replace(":", "_")

                    # Convert instance from Multi-SWE-bench format to SWE-bench Verified format (plus language column).
                    # Based on: https://github.com/OpenHands/OpenHands/blob/0.62.0/evaluation/benchmarks/multi_swe_bench/scripts/data/data_change.py
                    issue = input_row["resolved_issues"][0]
                    output_row = {
                        "instance_id": input_row["instance_id"],
                        "repo": input_row["org"] + "/" + input_row["repo"],
                        "language": convert_language(
                            data_file.parent.name if subset == "full" else input_row["language"]
                        ),
                        "base_commit": input_row["base"]["sha"],
                        "problem_statement": issue["title"] + "\n" + issue["body"],
                        "container_formatter": container_formatter.format(docker_image=docker_image),
                        "container_id": i,
                        "dataset_name": DATASET_NAME[subset],
                        "split": "train",
                        # We save the original row because the evaluation harness will need it.
                        # We save it as a string to prevent HF load_dataset errors.
                        "original_row": json.dumps(input_row),
                    }

                    print(json.dumps(output_row), file=fout)

    shutil.rmtree(temp_dir)
