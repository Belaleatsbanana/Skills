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

import argparse
import os
import subprocess
import textwrap
from shlex import join


def _start_gpu_keepalive(interval: int = 5) -> subprocess.Popen | None:
    """Launch a background process that keeps all GPUs active for DCGM monitoring.

    Uses one thread per GPU running sustained matmuls for WORK_SEC seconds,
    then sleeping SLEEP_SEC seconds.  This produces ~30-40% SM utilization on
    each GPU — clearly visible to DCGM and well above the idle-reaper threshold.
    """
    script = textwrap.dedent("""\
        import time, torch, os, signal, sys, threading
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
        n = torch.cuda.device_count()
        WORK = 2.0
        SLEEP = {interval}
        def worker(gid):
            dev = torch.device(f'cuda:{{gid}}')
            a = torch.randn(1024, 1024, device=dev)
            b = torch.randn(1024, 1024, device=dev)
            while True:
                try:
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < WORK:
                        torch.mm(a, b, out=a)
                    torch.cuda.synchronize(dev)
                except Exception as e:
                    print(f'gpu_keepalive GPU {{gid}}: {{e}}', flush=True)
                time.sleep(SLEEP)
        for i in range(n):
            threading.Thread(target=worker, args=(i,), daemon=True).start()
        print(f'gpu_keepalive: {{n}} GPU threads, work={{WORK}}s sleep={{SLEEP}}s (pid={{os.getpid()}})', flush=True)
        signal.pause()
    """).format(interval=interval)
    proc = subprocess.Popen(["python3", "-c", script])
    return proc


def main():
    parser = argparse.ArgumentParser(description="Serve vLLM model")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--no_verbose", action="store_true", help="Print verbose logs")
    args, unknown = parser.parse_known_args()

    extra_arguments = join(unknown)

    keepalive_proc = None
    if os.environ.get("GPU_KEEPALIVE", "0") == "1":
        interval = int(os.environ.get("GPU_KEEPALIVE_INTERVAL", "20"))
        keepalive_proc = _start_gpu_keepalive(interval)

    print(f"Deploying model {args.model}")
    print("Starting OpenAI Server")

    if args.no_verbose:
        logging_args = " --disable-log-requests --disable-log-stats "
    else:
        logging_args = ""

    cmd = (
        f"python3 -m vllm.entrypoints.openai.api_server "
        f'    --model="{args.model}" '
        f'    --served-model-name="{args.model}"'
        f"    --trust-remote-code "
        f'    --host="0.0.0.0" '
        f"    --port={args.port} "
        f"    --tensor-parallel-size={args.num_gpus * args.num_nodes} "  # TODO: is this a good default for multinode setup?
        f"    {logging_args} "
        f"    {extra_arguments} " + (' | grep -v "200 OK"' if args.no_verbose else "")
    )

    try:
        subprocess.run(cmd, shell=True, check=True)
    finally:
        if keepalive_proc is not None:
            keepalive_proc.terminate()
            keepalive_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
