# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

METRICS_TYPE = "multichoice"

# 5-shot only: resolve few-shot set per-row via `examples_type` field from dataset entries.
# Escaped quotes keep `{examples_type}` as a string for Hydra parsing, then prompt code formats it per row.
GENERATION_ARGS = "++prompt_config=multilingual/mmlu-prox ++eval_type=multichoice"
