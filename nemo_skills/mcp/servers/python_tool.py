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

import argparse
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict

from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.tool_providers import MCPClientTool
from nemo_skills.mcp.utils import add_config_args, load_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    output_dict: Dict[str, str]
    session_id: Any | None  # uuid


mcp = FastMCP(name="python_tool")

# Initialized from config in main()
sandbox = None

# TODO: how should we control timeout in description?
description = (
    "Call this function to execute Python code in a stateful Jupyter notebook environment. "
    "Python will respond with the output of the execution or time out after 120.0 seconds."
)


@mcp.tool(name="stateful_python_code_exec", description=description)
async def stateful_python_code_exec(
    code: Annotated[str, Field(description="Code to execute")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = 10,
    trace_id: Annotated[str | None, Field(description="Trace ID for call correlation")] = None,
) -> ExecutionResult:
    language = "ipython"

    if trace_id:
        logger.info(f"[TRACE:{trace_id}] layer=python_tool_server event=received")

    start_time = time.perf_counter()
    try:
        output_dict, session_id = await sandbox.execute_code(
            code, language=language, timeout=timeout, session_id=session_id, trace_id=trace_id
        )
    except RemoteProtocolError:
        output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
        session_id = None
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if trace_id:
            logger.info(f"[TRACE:{trace_id}] layer=python_tool_server event=done elapsed_ms={elapsed_ms:.1f}")

    return {"output_dict": output_dict, "session_id": session_id}


def main():
    parser = argparse.ArgumentParser(description="MCP HTTP server for executing Python code in a sandbox")
    # HTTP server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="Port for the HTTP server")
    # Sandbox configuration - CLI args take precedence over config file
    parser.add_argument("--sandbox-host", default=None, help="Sandbox server host (overrides config)")
    parser.add_argument("--sandbox-port", default=None, help="Sandbox server port (overrides config)")
    # Sandbox configuration via OmegaConf/Hydra (optional)
    add_config_args(parser)
    args = parser.parse_args()

    try:
        cfg = load_mcp_config(
            config=args.config,
            config_dir=args.config_dir,
            config_name=args.config_name,
        )
    except ValueError as e:
        logger.warning(f"{e} Falling back to default local sandbox config.")
        cfg = OmegaConf.create({"sandbox": {"sandbox_type": "local"}})

    global sandbox
    sandbox_cfg = OmegaConf.to_container(cfg.sandbox, resolve=True)

    # CLI args override config file values
    if args.sandbox_host:
        sandbox_cfg["host"] = args.sandbox_host
    if args.sandbox_port:
        sandbox_cfg["port"] = args.sandbox_port

    sandbox = get_sandbox(**sandbox_cfg)

    logger.info(f"Starting python_tool HTTP server on {args.host}:{args.port}")
    logger.info(f"Sandbox config: {sandbox_cfg}")

    # Reduce verbosity of MCP and httpx loggers (they log every request at INFO)
    for noisy_logger in [
        "mcp.server.streamable_http_manager",
        "mcp.server.streamable_http",
        "mcp.server.lowlevel.server",
        "mcp.server",
        "mcp.client.streamable_http",
        "httpx",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Run as HTTP server using uvicorn (fixes MCP stdio hang bug)
    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ==============================
# Module-based tool implementation
# ==============================


class PythonTool(MCPClientTool):
    # Default URL for the python_tool HTTP server (spawned by ns_tools)
    DEFAULT_BASE_URL = "http://127.0.0.1:8765/mcp"

    def __init__(self, base_url: str | None = None) -> None:
        super().__init__()
        # Use HTTP client to connect to python_tool server
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStreamableHttpClient",
                "client_params": {
                    "base_url": base_url or self.DEFAULT_BASE_URL,
                },
                # hide args from schemas and sanitize at runtime
                "hide_args": {"stateful_python_code_exec": ["session_id", "timeout", "trace_id"]},
                # execution-specific default
                "exec_timeout_s": 10,
            }
        )
        self.requests_to_sessions = defaultdict(lambda: None)

    async def execute(self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        # Ensure timeout is sent via extra_args (post-sanitize), not in main arguments
        arguments = dict(arguments)
        extra_args = dict(extra_args or {})
        request_id = extra_args.pop("request_id")
        trace_id = extra_args.get("trace_id")  # Keep trace_id in extra_args for passing through

        merged_extra = dict(extra_args)
        merged_extra.setdefault("timeout", self._config.get("exec_timeout_s", 10))
        merged_extra["session_id"] = self.requests_to_sessions[request_id]

        if trace_id:
            logger.info(f"[TRACE:{trace_id}] layer=python_tool event=mcp_call_start")

        start_time = time.perf_counter()
        try:
            result = await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if trace_id:
                logger.info(f"[TRACE:{trace_id}] layer=python_tool event=mcp_call_done elapsed_ms={elapsed_ms:.1f}")

        # Handle error responses that don't contain session_id
        if "error" in result:
            logger.warning(f"Tool execution error: {result['error']}")
            return f"Error: {result['error']}"

        # Update session mapping for stateful execution
        self.requests_to_sessions[request_id] = result.get("session_id")
        output = f"{result['output_dict']['stdout']}{result['output_dict']['stderr']}"
        if output.endswith("\n"):  # there is always a trailing newline, removing it
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        """Close the connection pool."""
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()


if __name__ == "__main__":
    main()
