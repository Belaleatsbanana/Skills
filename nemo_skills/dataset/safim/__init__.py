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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "code"
METRICS_TYPE = "infilling"
# Pipeline default input: nemo_skills/dataset/safim/{EVAL_SPLIT}.jsonl (see prepare.py).
# Use split=api|block|control when launching eval for each subset.
EVAL_SPLIT = "api"
# ``eval_config.subset`` must match the HuggingFace subset passed to ``safim.evaluate``.
GENERATION_ARGS = f"++prompt_config=generic/fim ++eval_type=safim ++eval_config.subset={EVAL_SPLIT}"
REQUIRES_SANDBOX = True
