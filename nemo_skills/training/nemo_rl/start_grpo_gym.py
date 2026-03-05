#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
GRPO training with NemoGym integration.
Supports both generic math and proof (theorem-proving) data formats,
as well as synchronous and asynchronous GRPO training.
"""

import argparse
import json
import os
import pprint
from itertools import chain, repeat
from typing import Optional

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _should_use_nemo_gym,
    grpo_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    nemo_gym_example_to_nemo_rl_datum_spec,
    setup_nemo_gym_config,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf

from nemo_skills.utils import setup_make_sequence_length_divisible_by

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

PROVER_PROMPT_TEMPLATE = """Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

```txt
Here is the instruction to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0

Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1
```

In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.
- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution
... // Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.

## Self Evaluation

Here is my evaluation of the solution:
... // Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.

Based on my evaluation, the final overall score should be:
\\boxed{{...}} // where ... should be the final overall score (0, 0.5, or 1, and nothing else) based on the evaluation instruction above. You should reach this score ONLY AFTER careful RE-examination of your own solution above

---

Here is your task input:

## Problem
{problem}"""


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with NemoGym")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args, overrides = parser.parse_known_args()
    return args, overrides


def load_proof_jsonl_to_nemo_gym_examples(
    jsonl_fpath: str,
    agent_name: str,
    prover_prompt_template: str,
    num_repeats: Optional[int] = None,
) -> list[dict]:
    """Load proof JSONL ({"problem": ...}) and convert to NemoGym format
    with responses_create_params and agent_ref."""
    with open(jsonl_fpath) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    examples = []
    for row in rows:
        problem = row.get("problem", "")
        user_content = prover_prompt_template.format(problem=problem)
        examples.append({
            "agent_ref": {"name": agent_name},
            "responses_create_params": {
                "input": [{"role": "user", "content": user_content}],
            },
            "problem": problem,
        })

    if num_repeats:
        examples = list(chain.from_iterable(repeat(ex, num_repeats) for ex in examples))

    return examples


def setup_single_nemo_gym_dataset(jsonl_fpath: str, tokenizer, num_repeats: Optional[int] = None):
    """Setup NemoGym dataset from pre-formatted JSONL (already has responses_create_params)."""
    with open(jsonl_fpath) as f:
        nemo_gym_examples = list(map(json.loads, f))

    print(f"Loaded data at {jsonl_fpath}. Found {len(nemo_gym_examples)} examples")

    if num_repeats:
        previous_length = len(nemo_gym_examples)
        nemo_gym_examples = list(
            chain.from_iterable(repeat(nemo_gym_example, num_repeats) for nemo_gym_example in nemo_gym_examples)
        )
        print(
            f"Repeating examples (in a pattern of abc to aabbcc) for {jsonl_fpath} from {previous_length} to {len(nemo_gym_examples)}!"
        )

    nemo_rl_compatible_examples: list[DatumSpec] = [
        nemo_gym_example_to_nemo_rl_datum_spec(nemo_gym_example, idx)
        for idx, nemo_gym_example in enumerate(nemo_gym_examples)
    ]

    def passthrough_task_processor(datum_dict, *args, **kwargs):
        return datum_dict

    return AllTaskProcessedDataset(
        nemo_rl_compatible_examples,
        tokenizer,
        None,
        passthrough_task_processor,
    )


def setup_proof_nemo_gym_dataset(
    jsonl_fpath: str,
    tokenizer,
    agent_name: str = "proof_simple_agent",
    prover_prompt_template: str = PROVER_PROMPT_TEMPLATE,
    num_repeats: Optional[int] = None,
):
    """Setup NemoGym dataset from proof JSONL ({"problem": ...}),
    applying the prover prompt template and creating responses_create_params."""
    nemo_gym_examples = load_proof_jsonl_to_nemo_gym_examples(
        jsonl_fpath, agent_name, prover_prompt_template, num_repeats
    )
    print(f"Loaded {len(nemo_gym_examples)} proof examples from {jsonl_fpath}")

    nemo_rl_compatible = [
        nemo_gym_example_to_nemo_rl_datum_spec(ex, idx)
        for idx, ex in enumerate(nemo_gym_examples)
    ]
    passthrough_task_processor = lambda datum_dict, *args, **kwargs: datum_dict
    return AllTaskProcessedDataset(
        nemo_rl_compatible,
        tokenizer,
        None,
        passthrough_task_processor,
    )


def main() -> None:
    """Main entry point."""
    args, overrides = parse_args()

    # Default config path
    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_gym.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    # Setup make_sequence_length_divisible_by
    if config["policy"]["make_sequence_length_divisible_by"] is None:
        tp = config["policy"]["tensor_model_parallel_size"]
        cp = config["policy"]["context_parallel_size"]
        config["policy"]["make_sequence_length_divisible_by"] = setup_make_sequence_length_divisible_by(tp, cp)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    # Print final config
    print("\nFinal config:")
    pprint.pprint(config)

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config)

    data_cfg = config["data"]
    agent_name = data_cfg.get("proof_agent_name", "proof_simple_agent")
    use_proof_format = data_cfg.get("proof_format", True)

    print("\n▶ Setting up data...")
    if use_proof_format:
        train_dataset = setup_proof_nemo_gym_dataset(
            jsonl_fpath=data_cfg["train_jsonl_fpath"],
            tokenizer=tokenizer,
            agent_name=agent_name,
            num_repeats=data_cfg.get("train_num_repeats"),
        )
        val_dataset = setup_proof_nemo_gym_dataset(
            jsonl_fpath=data_cfg["validation_jsonl_fpath"],
            tokenizer=tokenizer,
            agent_name=agent_name,
            num_repeats=data_cfg.get("validation_num_repeats"),
        )
    else:
        train_dataset = setup_single_nemo_gym_dataset(
            jsonl_fpath=data_cfg["train_jsonl_fpath"],
            tokenizer=tokenizer,
        )
        val_dataset = setup_single_nemo_gym_dataset(
            jsonl_fpath=data_cfg["validation_jsonl_fpath"],
            tokenizer=tokenizer,
        )

    config["grpo"]["max_val_samples"] = len(val_dataset)
    config["grpo"]["val_batch_size"] = len(val_dataset)

    print("Final config (partial):")
    pprint.pprint({k: config[k] for k in ["data", "env"] if k in config})

    init_ray()

    config.get("env", {}).get("nemo_gym", {}).pop("is_trajectory_collection", None)

    (
        policy,
        policy_generation,
        nemo_gym_env,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    task_to_env = {"nemo_gym": nemo_gym_env}
    val_task_to_env = task_to_env

    if config["grpo"].get("async_grpo", {}).get("enabled"):
        from nemo_rl.algorithms.grpo import async_grpo_train
        print("\n🚀 Running async GRPO training...")
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=config["grpo"]["async_grpo"]["max_trajectory_age_steps"],
        )
    else:
        print("\n🚀 Running synchronous GRPO training...")
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
