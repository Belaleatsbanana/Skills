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
METRICS_TYPE = "livecodebench_pro"
EVAL_SPLIT = "test_25q2"
# Grading uses go-judge over HTTP, not the NeMo sandbox container. Run go-judge where cgroups work
# (e.g. separate host/VM with Docker --privileged) and set NEMO_SKILLS_GO_JUDGE_HOST / _PORT, or use 127.0.0.1
# if go-judge runs on the same node as the eval step.
REQUIRES_SANDBOX = False
# Prepended to generation args. Host: os.environ NEMO_SKILLS_GO_JUDGE_HOST or Hydra go_judge.host, else 127.0.0.1.
EVAL_ARGS = "++eval_config.go_judge.port=5050 ++eval_config.go_judge.http_timeout=120.0"
GENERATION_ARGS = "++prompt_config=eval/livecodebench/default_reasoning ++eval_type=livecodebench_pro"
