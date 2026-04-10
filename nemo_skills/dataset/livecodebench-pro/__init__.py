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
# Grading uses go-judge over HTTP. The main nemo-skills image includes /usr/local/bin/go-judge; eval runs it
# locally before generation/eval unless NEMO_SKILLS_SKIP_AUTO_GO_JUDGE=1 or you use an external judge
# (NEMO_SKILLS_GO_JUDGE_HOST / ++eval_config.go_judge.host).
REQUIRES_SANDBOX = False
# Shell snippet for the main Slurm step only (merged with ns eval --installation-command). Skip auto-start: NEMO_SKILLS_SKIP_AUTO_GO_JUDGE=1
INSTALLATION_COMMAND = (
    r'bash -lc "set -e; [ \"${NEMO_SKILLS_SKIP_AUTO_GO_JUDGE:-0}\" = \"1\" ] && exit 0; '
    r"case \"${NEMO_SKILLS_GO_JUDGE_HOST:-}\" in \"\"|127.0.0.1|localhost) ;; *) echo \"[nemo-skills] NEMO_SKILLS_GO_JUDGE_HOST=${NEMO_SKILLS_GO_JUDGE_HOST} (non-local); skip auto go-judge\" >&2; exit 0 ;; esac; "
    r"test -x /usr/local/bin/go-judge || { echo \"[nemo-skills] FATAL: /usr/local/bin/go-judge missing. Point containers.nemo-skills at an image built from Dockerfile.nemo-skills (with go-judge), or set NEMO_SKILLS_GO_JUDGE_HOST to an external judge and NEMO_SKILLS_SKIP_AUTO_GO_JUDGE=1\" >&2; exit 1; }; "
    r"rm -f /tmp/go-judge.log; touch /tmp/go-judge.log; "
    r"nohup /usr/local/bin/go-judge -http-addr 127.0.0.1:${NEMO_SKILLS_GO_JUDGE_PORT:-5050} >>/tmp/go-judge.log 2>&1 & "
    r"sleep 2; for i in 1 2 3 4 5 6 7 8 9 10; do curl -sf --connect-timeout 1 \"http://127.0.0.1:${NEMO_SKILLS_GO_JUDGE_PORT:-5050}/version\" >/dev/null && exit 0; sleep 1; done; "
    r'echo \"[nemo-skills] FATAL: go-judge did not become ready on 127.0.0.1:${NEMO_SKILLS_GO_JUDGE_PORT:-5050}. Last log:\" >&2; tail -50 /tmp/go-judge.log >&2; exit 1"'
)
# Prepended to generation args. Host: os.environ NEMO_SKILLS_GO_JUDGE_HOST or Hydra go_judge.host, else 127.0.0.1.
EVAL_ARGS = "++eval_config.go_judge.port=5050 ++eval_config.go_judge.http_timeout=120.0"
GENERATION_ARGS = "++prompt_config=eval/livecodebench/default_reasoning ++eval_type=livecodebench_pro"
