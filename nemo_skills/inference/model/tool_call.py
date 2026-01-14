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

import asyncio
import copy
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Dict, List

from nemo_skills.mcp.adapters import (
    format_tool_list_by_endpoint_type,
    format_tool_response_by_endpoint_type,
    get_tool_details_by_endpoint_type,
    load_schema_overrides,
    remap_tool_call,
)
from nemo_skills.mcp.tool_manager import FatalToolError, ToolManager
from nemo_skills.utils import get_logger_name

from .base import BaseModel, EndpointType

LOG = logging.getLogger(get_logger_name(__file__))


class ToolCallingWrapper:
    """
    Wrapper to handle tool calling.

    TODO(sanyamk): Supports only Chat Completions API for now.
    """

    def __init__(
        self,
        model: BaseModel,
        tool_modules: list[str] | None = None,
        tool_overrides: dict | None = None,
        additional_config: dict | None = None,
        schema_overrides: dict | None = None,
    ):
        self.model = model
        additional_config = additional_config or {}

        assert tool_modules, "tool_modules must be provided for tool calling"
        self.tool_manager = ToolManager(
            module_specs=tool_modules,
            overrides=tool_overrides or {},
            context=additional_config,
        )

        self.schema_overrides = load_schema_overrides(schema_overrides)
        self.schema_mappings = {}  # Built when tools are listed

    async def _execute_tool_call(self, tool_call, request_id: str, endpoint_type: EndpointType):
        """Execute a single tool call and return result with timing info.

        Returns:
            tuple: (result, timing_info) where timing_info is a dict with:
                - tool_name: Name of the tool called
                - start_time: Unix timestamp when execution started
                - end_time: Unix timestamp when execution ended
                - duration_ms: Duration in milliseconds
        """
        ## TODO(sanyamk): The correct key format needs to be cohesive with other formatters.
        tool_name, tool_args = get_tool_details_by_endpoint_type(tool_call, endpoint_type)

        ##
        # TODO(sanyamk): Not all tool arguments might necessarily be in JSON format.
        #   Kept here to handle errors for now.

        try:
            tool_args = json.loads(tool_args)
        except json.decoder.JSONDecodeError as e:
            LOG.error(f"Tool arguments are not in JSON format: {tool_args}")
            LOG.exception(e)
            return {"error": "Tool argument parsing failed."}, {
                "tool_name": tool_name,
                "start_time": time.time(),
                "end_time": time.time(),
                "duration_ms": 0.0,
                "error": True,
            }

        ## TODO(sanyamk): Only exceptions related to tool execution here, all others must fail.
        # Remap model's tool name/args back to original schema
        original_tool_name, tool_args = remap_tool_call(tool_name, tool_args, self.schema_mappings)

        start_time = time.time()
        try:
            # Allow providers to specify extra_args behavior internally if needed in the future
            result = await self.tool_manager.execute_tool(
                original_tool_name, tool_args, extra_args={"request_id": request_id}
            )
            end_time = time.time()
            timing_info = {
                "tool_name": original_tool_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": (end_time - start_time) * 1000,
                "error": False,
            }
        except FatalToolError:
            # Fatal errors should propagate up and stop the process
            raise
        except Exception as e:
            LOG.exception(e)
            end_time = time.time()
            return {"error": "Tool execution failed."}, {
                "tool_name": original_tool_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": (end_time - start_time) * 1000,
                "error": True,
            }

        return result, timing_info

    async def _execute_tool_calls(self, tool_calls: List, request_id: str, endpoint_type: EndpointType):
        """Execute multiple tool calls in parallel and return results with timing.

        Returns:
            tuple: (formatted_responses, timing_infos) where timing_infos is a list of
                   timing info dicts from each tool call.
        """
        tasks = [
            self._execute_tool_call(tool_call, request_id=request_id, endpoint_type=endpoint_type)
            for tool_call in tool_calls
        ]
        results_with_timing = await asyncio.gather(*tasks)

        # Unpack results and timing info
        tool_results = [r[0] for r in results_with_timing]
        timing_infos = [r[1] for r in results_with_timing]

        formatted_responses = [
            format_tool_response_by_endpoint_type(tool_call, tool_result, endpoint_type)
            for tool_call, tool_result in zip(tool_calls, tool_results)
        ]

        return formatted_responses, timing_infos

    async def generate_async(
        self,
        prompt: List,
        endpoint_type: EndpointType,
        tools: List[dict] = None,
        tokens_to_generate: int = None,
        **generation_kwargs,
    ) -> Dict:
        assert isinstance(prompt, list), "Only use ChatCompletion API for now."

        assert tools is None, "Do not pass 'tools'; they are derived from tool_modules."

        # This assumes that the available tools do not change during the generation.
        raw_tools = await self.tool_manager.list_all_tools(use_cache=True)
        tools, self.schema_mappings = format_tool_list_by_endpoint_type(
            raw_tools, endpoint_type, schema_overrides=self.schema_overrides
        )
        LOG.info("Available Tools: %s", tools)

        result_steps = defaultdict(list)
        conversation = copy.deepcopy(prompt)

        # Track tool call timing
        all_tool_call_timings = []

        # assigning a unique request id to pass to tool calls if they need to be stateful
        request_id = str(uuid.uuid4())

        while True:
            if isinstance(tokens_to_generate, int) and tokens_to_generate <= 0:
                break
            generation = await self.model.generate_async(
                prompt=conversation,
                tools=tools,
                tokens_to_generate=tokens_to_generate,
                endpoint_type=endpoint_type,
                **generation_kwargs,
            )
            if isinstance(tokens_to_generate, int):
                tokens_to_generate -= generation["num_generated_tokens"]

            for k in ["generation", "num_generated_tokens", "reasoning_content", "finish_reason"]:
                if k in generation:
                    result_steps[k].append(generation[k])

            conversation.extend(generation["serialized_output"])

            tool_calls = generation.get("tool_calls", [])
            if tool_calls:
                tool_calls = [tool_call.model_dump() for tool_call in tool_calls]
                tool_calls_output_messages, timing_infos = await self._execute_tool_calls(
                    tool_calls, request_id=request_id, endpoint_type=endpoint_type
                )
                LOG.info("Sending tool calls: %s", tool_calls_output_messages)
                conversation.extend(tool_calls_output_messages)

                # Collect timing info
                all_tool_call_timings.extend(timing_infos)
                for timing in timing_infos:
                    LOG.info(
                        "Tool call timing: %s took %.2f ms",
                        timing["tool_name"],
                        timing["duration_ms"],
                    )

                result_steps["num_tool_calls"].append(len(tool_calls))

                continue

            break

        result_steps["generation"] = "".join(result_steps["generation"])
        result_steps["num_generated_tokens"] = sum(result_steps["num_generated_tokens"])
        result_steps["num_tool_calls"] = sum(result_steps["num_tool_calls"])
        result_steps["conversation"] = conversation
        result_steps["tools"] = tools  # Schema sent to model (with overrides applied)

        # Add tool call timing statistics
        if all_tool_call_timings:
            durations = [t["duration_ms"] for t in all_tool_call_timings]
            result_steps["tool_call_times"] = all_tool_call_timings
            result_steps["total_tool_call_time_ms"] = sum(durations)
            result_steps["avg_tool_call_time_ms"] = sum(durations) / len(durations)
            result_steps["min_tool_call_time_ms"] = min(durations)
            result_steps["max_tool_call_time_ms"] = max(durations)
        else:
            result_steps["tool_call_times"] = []
            result_steps["total_tool_call_time_ms"] = 0.0
            result_steps["avg_tool_call_time_ms"] = 0.0
            result_steps["min_tool_call_time_ms"] = 0.0
            result_steps["max_tool_call_time_ms"] = 0.0

        return result_steps
