# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    api_dataset = load_dataset("gonglinyuan/safim", "api", split="test")
    block_dataset = load_dataset("gonglinyuan/safim", "block", split="test")
    control_dataset = load_dataset("gonglinyuan/safim", "control", split="test")

    api_dataset = api_dataset.map(
        lambda x: {
            "prefix": x["eval_prompt"].split("{{completion}}")[0],
            "suffix": x["eval_prompt"].split("{{completion}}")[1],
        }
    )
    block_dataset = block_dataset.map(
        lambda x: {
            "prefix": x["eval_prompt"].split("{{completion}}")[0],
            "suffix": x["eval_prompt"].split("{{completion}}")[1],
        }
    )
    control_dataset = control_dataset.map(
        lambda x: {
            "prefix": x["eval_prompt"].split("{{completion}}")[0],
            "suffix": x["eval_prompt"].split("{{completion}}")[1],
        }
    )

    api_dataset = api_dataset.map(lambda x: {"language": x["lang"]})
    block_dataset = block_dataset.map(lambda x: {"language": x["lang"]})
    control_dataset = control_dataset.map(lambda x: {"language": x["lang"]})

    api_dataset = api_dataset.remove_columns(["prompt", "eval_prompt", "lang"])
    block_dataset = block_dataset.remove_columns(["prompt", "eval_prompt", "lang"])
    control_dataset = control_dataset.remove_columns(["prompt", "eval_prompt", "lang"])

    print(f"Number of examples in api: {len(api_dataset)}")
    print(f"Number of examples in block: {len(block_dataset)}")
    print(f"Number of examples in control: {len(control_dataset)}")

    api_dataset.to_json("api.jsonl", orient="records", lines=True)
    block_dataset.to_json("block.jsonl", orient="records", lines=True)
    control_dataset.to_json("control.jsonl", orient="records", lines=True)
