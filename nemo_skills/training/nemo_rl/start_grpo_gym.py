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
Standard flow: data is pre-formatted by Gym's prepare_data.py (proof_with_judge env).
Supports synchronous and asynchronous GRPO training.
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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with NemoGym")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_single_nemo_gym_dataset(jsonl_fpath: str, tokenizer, num_repeats: Optional[int] = None):
    """Setup NemoGym dataset from pre-formatted JSONL (already has responses_create_params).

    Data should be prepared by Gym's prepare_data.py beforehand.
    """
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

    if not os.environ.get("PROOF_JUDGE_LOG_JSONL_PATH"):
        os.environ["PROOF_JUDGE_LOG_JSONL_PATH"] = os.path.join(
            config["logger"]["log_dir"], "proof_judge_log.jsonl"
        )
    print(f"📝 proof_judge JSONL log: {os.environ['PROOF_JUDGE_LOG_JSONL_PATH']}")

    print("\nFinal config:")
    pprint.pprint(config)

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(config["policy"]["generation"], tokenizer)

    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config)

    data_cfg = config["data"]

    print("\n▶ Setting up data...")
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
