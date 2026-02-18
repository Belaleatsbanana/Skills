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

# Few-shot toggle:
# - False (default): 0-shot using generic/default
# - True: 5-shot using per-row examples_type from dataset rows
ENABLE_FEW_SHOT = True

GENERATION_ARGS = (
    "++prompt_config=generic/general-boxed ++examples_type='{examples_type}' ++eval_type=multichoice"
    if ENABLE_FEW_SHOT
    else "++prompt_config=generic/default ++eval_type=multichoice"
)
