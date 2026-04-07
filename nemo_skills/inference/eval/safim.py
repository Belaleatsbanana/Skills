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

"""SAFIM fill-in-the-middle generation entrypoint.

Each JSONL record is passed through to ``generic/fim``. The template expects
``language``, ``prefix``, ``suffix``, and ``comment_delimiter`` (as produced by
``dataset/safim/prepare.py``). Extra columns are fine: ``str.format`` ignores
unused keys.

Pipeline data paths use ``EVAL_SPLIT`` in ``dataset/safim/__init__.py`` (default
``api``), i.e. ``.../safim/api.jsonl``. Use split ``block`` or ``control`` for
the other prepared files.

``eval_type`` defaults to ``None`` here so ad-hoc runs work without an evaluator;
the dataset's ``GENERATION_ARGS`` adds ``++eval_type=safim`` for full pipelines,
which requires registering ``safim`` in ``EVALUATOR_MAP`` / class map.

Example::

    python -m nemo_skills.inference.eval.safim \\
        input_file=.../api.jsonl output_file=.../out.jsonl
"""

import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class SafimGenerationConfig(GenerateSolutionsConfig):
    """SAFIM benchmark: one LLM call per row using the generic FIM prompt.

    For the full list of supported parameters, use
    ``python -m nemo_skills.inference.eval.safim --help``.
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    prompt_config: str = "generic/fim"
    code_execution: bool = False

    # Pipeline / dataset defaults use ``++eval_type=safim`` once the evaluator is registered.
    # Keep ``None`` for generation-only runs until ``safim`` exists in ``EVALUATOR_MAP``.
    eval_type: str | None = None
    eval_config: dict = field(default_factory=dict)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_safim_generation_config", node=SafimGenerationConfig)


class SafimGenerationTask(GenerationTask):
    """Thin wrapper for logging; default ``process_single_datapoint`` fills the FIM prompt from the row."""

    def log_example_prompt(self, data):
        if not data:
            return
        sample = data[0]
        LOG.info(
            "Example FIM prompt (first row keys=%s):\n%s",
            list(sample.keys()),
            self.fill_prompt(sample, data),
        )


GENERATION_TASK_CLASS = SafimGenerationTask


@hydra.main(version_base=None, config_name="base_safim_generation_config")
def safim_generation(cfg: SafimGenerationConfig):
    cfg = SafimGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SafimGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SafimGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        safim_generation()
