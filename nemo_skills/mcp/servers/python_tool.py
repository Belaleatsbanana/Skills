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
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List

import httpx
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.tool_manager import Tool
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
) -> ExecutionResult:
    language = "ipython"
    try:
        output_dict, session_id = await sandbox.execute_code(
            code, language=language, timeout=timeout, session_id=session_id
        )
    except httpx.RemoteProtocolError:
        output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
        session_id = None

    return {"output_dict": output_dict, "session_id": session_id}


def main():
    parser = argparse.ArgumentParser(description="MCP server for executing Python code in a sandbox")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP server to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the HTTP server to")
    parser.add_argument(
        "--disable-session-restore",
        action="store_true",
        default=False,
        help="Skip replaying session history after sandbox worker restarts (overrides config)",
    )
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
    if args.disable_session_restore:
        sandbox_cfg["disable_session_restore"] = True

    sandbox = get_sandbox(**sandbox_cfg)
    mcp.run(transport="streamable-http", host=args.host, port=args.port)


# ==============================
# Module-based tool implementation
# ==============================


def _get_free_port():
    """Get a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class PythonTool(MCPClientTool):
    """Tool provider that spawns a persistent python_tool HTTP server and connects via MCP Streamable HTTP."""

    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                # placeholder base_url — replaced in post_configure() after server starts
                "client": "nemo_skills.mcp.clients.MCPStreamableHttpClient",
                "client_params": {"base_url": "http://127.0.0.1:0/mcp"},
                "hide_args": {"stateful_python_code_exec": ["session_id", "timeout"]},
                "init_hook": "hydra",
                "exec_timeout_s": 10,
                "server_host": "127.0.0.1",
                "server_port": 0,  # 0 = auto-allocate
            }
        )
        self._server_process = None
        self._config_tmpfile = None
        self.requests_to_sessions = defaultdict(lambda: None)

    def configure(self, overrides=None, context=None):
        self._context = context or {}
        super().configure(overrides, context)

    def post_configure(self):
        port = self._config.get("server_port", 0)
        if not port:
            port = _get_free_port()
        host = self._config.get("server_host", "127.0.0.1")

        sandbox_cfg = self._context.get("sandbox", {})
        self._start_server(host, port, sandbox_cfg)

        # Replace the placeholder client with one pointing at the running server
        from nemo_skills.mcp.clients import MCPStreamableHttpClient

        self._client = MCPStreamableHttpClient(base_url=f"http://{host}:{port}/mcp")
        self._client._hide_args = self._config.get("hide_args", {})
        self._client._disabled_tools = set(self._config.get("disabled_tools", []))
        self._client._enabled_tools = set(self._config.get("enabled_tools", []))

    def _start_server(self, host, port, sandbox_cfg):
        cfg = OmegaConf.create({"sandbox": sandbox_cfg or {"sandbox_type": "local"}})
        self._config_tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        OmegaConf.save(cfg, self._config_tmpfile)
        self._config_tmpfile.close()

        cmd = [
            sys.executable,
            "-m",
            "nemo_skills.mcp.servers.python_tool",
            "--host",
            host,
            "--port",
            str(port),
            "--config",
            self._config_tmpfile.name,
        ]
        logger.info(f"Starting python_tool HTTP server: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(cmd)
        self._wait_for_ready(host, port)
        logger.info(f"python_tool HTTP server ready (PID: {self._server_process.pid})")

    def _wait_for_ready(self, host, port, timeout=30):
        url = f"http://{host}:{port}/mcp"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    f"python_tool server exited during startup (code {self._server_process.returncode})"
                )
            try:
                resp = httpx.get(url, timeout=2)
                if resp.status_code < 500:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"python_tool server not ready after {timeout}s")

    async def execute(self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        arguments = dict(arguments)
        request_id = extra_args.pop("request_id")
        merged_extra = dict(extra_args or {})
        merged_extra.setdefault("timeout", self._config.get("exec_timeout_s", 10))
        merged_extra["session_id"] = self.requests_to_sessions[request_id]
        result = await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)
        self.requests_to_sessions[request_id] = result["session_id"]
        output = f"{result['output_dict']['stdout']}{result['output_dict']['stderr']}"
        if output.endswith("\n"):  # there is always a trailing newline, removing it
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        if self._server_process:
            logger.info(f"Terminating python_tool server (PID: {self._server_process.pid})")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("python_tool server did not terminate, killing...")
                self._server_process.kill()
            self._server_process = None
        if self._config_tmpfile:
            try:
                os.unlink(self._config_tmpfile.name)
            except OSError:
                pass
            self._config_tmpfile = None


class DirectPythonTool(Tool):
    """Python code execution tool that calls the sandbox directly, bypassing MCP.

    This is a drop-in replacement for PythonTool that eliminates the MCP protocol
    overhead (subprocess spawning, MCP session initialization, JSON-RPC serialization)
    by calling sandbox.execute_code() directly via HTTP.

    Shared config keys with PythonTool (so switching is just changing the module spec):
        - hide_args: controls which args are stripped from schemas and sanitized at runtime
        - exec_timeout_s: default execution timeout

    Usage:
        tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            # Same keys/defaults as PythonTool (minus MCP-specific: client, client_params, init_hook)
            "hide_args": {"stateful_python_code_exec": ["session_id", "timeout"]},
            "exec_timeout_s": 10,
            "sandbox": {},
        }
        self._sandbox = None
        self._sanitize_keys: Dict[str, set] = {}
        self.requests_to_sessions: Dict[str, Any] = defaultdict(lambda: None)

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        if overrides:
            self._config.update(overrides)

        # Build sanitize sets from hide_args (same source of truth as MCP path)
        hide_args = self._config.get("hide_args", {})
        self._sanitize_keys = {tool: set(keys) for tool, keys in hide_args.items()}

        # Build sandbox config from context (same source as the MCP server's main())
        sandbox_cfg = dict((context or {}).get("sandbox", {}))
        sandbox_cfg.update(self._config.get("sandbox", {}))
        sandbox_cfg.pop("sandbox_type", None)
        sandbox_type = (context or {}).get("sandbox", {}).get("sandbox_type", "local")
        sandbox_type = self._config.get("sandbox", {}).get("sandbox_type", sandbox_type)
        self._sandbox = get_sandbox(sandbox_type=sandbox_type, **sandbox_cfg)

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "stateful_python_code_exec",
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to execute"},
                    },
                    "required": ["code"],
                },
            }
        ]

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None
    ) -> str:
        # Strip model-supplied hidden args using hide_args config (same source as MCP sanitize())
        hidden = self._sanitize_keys.get(tool_name, set())
        arguments = {k: v for k, v in arguments.items() if k not in hidden}

        extra_args = dict(extra_args or {})
        request_id = extra_args.pop("request_id", None)
        timeout = extra_args.get("timeout", self._config.get("exec_timeout_s", 10))
        session_id = self.requests_to_sessions[request_id] if request_id is not None else None

        try:
            output_dict, session_id = await self._sandbox.execute_code(
                arguments["code"],
                language="ipython",
                timeout=timeout,
                session_id=session_id,
            )
        except RemoteProtocolError:
            output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
            session_id = None

        if request_id is not None:
            self.requests_to_sessions[request_id] = session_id

        output = f"{output_dict['stdout']}{output_dict['stderr']}"
        if output.endswith("\n"):
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.close()


if __name__ == "__main__":
    main()
