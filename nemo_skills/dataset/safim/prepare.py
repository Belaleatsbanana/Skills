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

from pathlib import Path

from datasets import load_dataset


def process_safim_subset(subset_name: str, output_dir: Path) -> int:
    """Loads and transforms a specific SAFIM subset."""
    print(f"Processing subset: {subset_name}...")

    # Load dataset
    ds = load_dataset("gonglinyuan/safim", subset_name, split="test")

    def transform(example):
        # 1. Split the prompt into prefix and suffix
        parts = example["eval_prompt"].split("{{completion}}")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        # 2. Map language to comment delimiter
        # Using a dictionary is much safer than complex and/or logic
        lang = str(example["lang"]).lower()
        delimiters = {
            "python": "#",
            "cpp": "//",
            "java": "//",
            "javascript": "//",
            "csharp": "//",
        }
        comment_delimiter = delimiters.get(lang, "#")

        return {
            "prefix": prefix,
            "suffix": suffix,
            "language": example["lang"],
            "subset_for_metrics": example["lang"],
            "comment_delimiter": comment_delimiter,
        }

    # Apply transformations and remove old columns in one go
    ds = ds.map(transform)
    ds = ds.remove_columns(["prompt", "eval_prompt", "lang"])

    output_path = output_dir / f"{subset_name}.jsonl"
    ds.to_json(str(output_path), orient="records", lines=True)

    return len(ds)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    subsets = ["api", "block", "control"]

    for subset in subsets:
        count = process_safim_subset(subset, data_dir)
        print(f"Number of examples in {subset}: {count}")
