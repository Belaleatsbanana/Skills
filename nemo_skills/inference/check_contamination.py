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

import asyncio
import json
import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class CheckContaminationConfig(GenerationTaskConfig):
    """LLM-based check contamination parameters.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    # Override the default Generation config here
    code_execution: bool = False
    prompt_config: str = "judge/check-contamination"
    generation_key: str = "contaminated"

    # Contamination-specific parameters
    retrieve_key: str = "problem"  # will be used to fill in prompt with retrieve_key1 and retrieve_key2
    # ask both with retrieve_key1 / retrieve_key2 and retrieve_key2 / retrieve_key1 and fill True if any is True
    check_both_ways: bool = False
    # Number of similar items to check. If not provided, will use the number of similar items in the first data point.
    top_k: int | None = None

    def _get_disallowed_params(self):
        """Returns a list of parameters with their default values to check that they are not changed from the defaults"""
        return [
            ("code_execution", False),
            ("sandbox", {}),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_check_contamination_config", node=CheckContaminationConfig)


class CheckContaminationTask(GenerationTask):
    def __init__(self, cfg: CheckContaminationConfig):
        super().__init__(cfg)

    async def postprocess_single_output(self, output, original_data_point):
        """Keep judgments and contaminated_per_item in output so they are written to the file."""
        extra = {k: output.get(k) for k in ("judgments", "contaminated_per_item", "all_generations") if k in output}
        await super().postprocess_single_output(output, original_data_point)
        for k, v in extra.items():
            if v is not None:
                output[k] = v

    def load_data(self):
        # Load the data as done in the base class
        data = super().load_data()

        # Adjust the batch size to account for the number of similar items
        if self.cfg.top_k is None:
            self.cfg.top_k = len(data[0]["similar_items"])

        return data

    def log_example_prompt(self, data):
        data_point = data[0]
        query_item = data_point[self.cfg.retrieve_key]
        similar_item = data_point["similar_items"][0]
        first_element = {
            f"{self.cfg.retrieve_key}1": query_item,
            f"{self.cfg.retrieve_key}2": similar_item,
        }
        LOG.info(
            "Example prompt:\nData dictionary: %s\nPrompt: %s",
            first_element,
            self.fill_prompt(first_element, data),
        )

    def _create_query_data(self, data_point):
        """Create query instances given the original instance"""
        query_data = []
        for similar_item in data_point["similar_items"][: self.cfg.top_k]:
            query_data.append(
                {
                    f"{self.cfg.retrieve_key}1": data_point[self.cfg.retrieve_key],
                    f"{self.cfg.retrieve_key}2": similar_item,
                }
            )

            if self.cfg.check_both_ways:
                query_data.append(
                    {
                        f"{self.cfg.retrieve_key}2": data_point[self.cfg.retrieve_key],
                        f"{self.cfg.retrieve_key}1": similar_item,
                    }
                )

        return query_data

    def prefill_generation(self, data_point):
        """Prefill disabled: always run full LLM judgment for all similar items."""
        return None

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single data point by running contamination checks on all similar items."""
        query_data = self._create_query_data(data_point)

        # Create tasks for all queries using super().process_single_datapoint
        tasks = []
        for query_point in query_data:
            tasks.append(super().process_single_datapoint(query_point, all_data))

        query_results = await asyncio.gather(*tasks)

        # Process results to determine if contaminated
        all_generations = []
        contaminated = False
        for result in query_results:
            generation = result["generation"]
            all_generations.append(generation)
            if generation.strip() == "True":
                contaminated = True

        # Build per-pair judgments for writing to output (one entry per similar_item)
        similar_items = data_point["similar_items"][: self.cfg.top_k]
        similarity_scores = data_point.get("similarity_scores")
        if similarity_scores is not None:
            similarity_scores = similarity_scores[: len(similar_items)]
        else:
            similarity_scores = [None] * len(similar_items)

        # When check_both_ways=True we have 2 results per similar_item (indices 2*i, 2*i+1)
        step = 2 if self.cfg.check_both_ways else 1

        judgments = []
        for i, similar_item in enumerate(similar_items):
            score = similarity_scores[i] if i < len(similarity_scores) else None
            idx = i * step
            if idx < len(all_generations):
                judged = any(
                    all_generations[idx + j].strip() == "True" for j in range(step) if idx + j < len(all_generations)
                )
                raw = [all_generations[idx + j] for j in range(step) if idx + j < len(all_generations)]
            else:
                judged = None
                raw = []
            judgments.append(
                {
                    "similar_item": similar_item,
                    "similarity_score": score,
                    "contaminated": judged,
                    "raw_generation": raw[0] if len(raw) == 1 else raw if raw else None,
                }
            )

        # Flat list of 20 True/False for quick inspection: contaminated_per_item[i] = judgment for similar_items[i]
        contaminated_per_item = [j["contaminated"] for j in judgments]

        return {
            "all_generations": all_generations,
            "generation": contaminated,
            "judgments": judgments,
            "contaminated_per_item": contaminated_per_item,
        }

    def postprocess(self):
        """Postprocess the output file to calculate the contamination portion."""
        num_contaminated, total = 0, 0
        with open(self.cfg.output_file, "r", encoding="utf-8", buffering=1) as fin:
            for line in fin:
                total += 1
                data_point = json.loads(line)
                if data_point[self.cfg.generation_key]:
                    num_contaminated += 1

        if total > 0:
            LOG.info("Contamination portion: %.2f%% (%d/%d)", 100 * num_contaminated / total, num_contaminated, total)


GENERATION_TASK_CLASS = CheckContaminationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_check_contamination_config")
def check_contamination(cfg: CheckContaminationConfig):
    cfg = CheckContaminationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = CheckContaminationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    CheckContaminationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()
