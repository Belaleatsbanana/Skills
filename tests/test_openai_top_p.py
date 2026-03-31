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

"""Unit tests for the top_p Optional[float] changes introduced in the PR.

Changes under test:
  - nemo_skills/inference/generate.py: InferenceConfig.top_p type changed to float | None
  - nemo_skills/inference/model/openai.py: _build_chat_request_params() updated to:
      * not include top_p in params when top_p is None (standard models)
      * not raise ValueError when top_p is None for reasoning models
      * still raise ValueError when top_p is a non-default, non-None value for reasoning models
"""

# Stub out heavy optional dependencies that are not installed in the unit-test
# environment before any nemo_skills package is imported.
import sys
import types
from unittest.mock import MagicMock

# latex2sympy2_extended and its transitive consumers
sys.modules.setdefault("latex2sympy2_extended", MagicMock())

_math_grader = types.ModuleType("nemo_skills.evaluation.math_grader")
_math_grader.extract_answer = MagicMock()
sys.modules.setdefault("nemo_skills.evaluation.math_grader", _math_grader)

# nemo_skills.dataset.utils is pulled in transitively via mcp.utils and code_execution
_ds_utils = types.ModuleType("nemo_skills.dataset.utils")
_ds_utils.locate = MagicMock()
_ds_utils.get_dataset_module = MagicMock()
_ds_utils.get_lean4_header = MagicMock()
sys.modules.setdefault("nemo_skills.dataset.utils", _ds_utils)

_mcp_utils = types.ModuleType("nemo_skills.mcp.utils")
_mcp_utils.locate = MagicMock()
sys.modules.setdefault("nemo_skills.mcp.utils", _mcp_utils)

# code_execution modules pulled in by the inference model package
sys.modules.setdefault("nemo_skills.code_execution.proof_utils", MagicMock())
sys.modules.setdefault("nemo_skills.code_execution.sandbox", MagicMock())

import pytest

from nemo_skills.inference.model.openai import OpenAIModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_model(model_name: str) -> OpenAIModel:
    """Create an OpenAIModel instance without calling __init__ (avoids API key / network setup)."""
    m = OpenAIModel.__new__(OpenAIModel)
    m.model = model_name
    # Prevent AttributeError in BaseModel.__del__ during GC (attribute set in __init__)
    m._tunnel = None
    return m


def _standard_chat_params(**overrides):
    """Return a minimal valid set of keyword arguments for _build_chat_request_params."""
    defaults = dict(
        messages=[{"role": "user", "content": "hi"}],
        tokens_to_generate=100,
        temperature=0.0,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        random_seed=0,
        stop_phrases=None,
        timeout=None,
        top_logprobs=None,
        stream=False,
        reasoning_effort=None,
        extra_body=None,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Tests: top_p=None with standard (non-reasoning) models
# ---------------------------------------------------------------------------


def test_build_chat_request_params_top_p_none_omitted_for_standard_model():
    """When top_p is None, it must NOT be included in the returned params for standard models."""
    model = _make_openai_model("gpt-4o")
    params = model._build_chat_request_params(**_standard_chat_params(top_p=None))
    assert "top_p" not in params


def test_build_chat_request_params_top_p_default_included_for_standard_model():
    """When top_p is 0.95 (default), it should be present in the returned params for standard models."""
    model = _make_openai_model("gpt-4o")
    params = model._build_chat_request_params(**_standard_chat_params(top_p=0.95))
    assert params["top_p"] == 0.95


def test_build_chat_request_params_top_p_custom_float_included_for_standard_model():
    """When top_p is a custom float, it should be forwarded in params for standard models."""
    model = _make_openai_model("gpt-4o")
    params = model._build_chat_request_params(**_standard_chat_params(top_p=0.7))
    assert params["top_p"] == 0.7


def test_build_chat_request_params_top_p_zero_included_for_standard_model():
    """top_p=0.0 is a valid boundary float and must be included in params (not treated as None)."""
    model = _make_openai_model("gpt-4o")
    params = model._build_chat_request_params(**_standard_chat_params(top_p=0.0))
    assert "top_p" in params
    assert params["top_p"] == 0.0


# ---------------------------------------------------------------------------
# Tests: top_p with reasoning models
# ---------------------------------------------------------------------------


def test_build_chat_request_params_reasoning_model_top_p_none_does_not_raise():
    """top_p=None must NOT raise a ValueError for reasoning models (new behaviour after PR)."""
    model = _make_openai_model("o1")
    # Should not raise
    params = model._build_chat_request_params(**_standard_chat_params(top_p=None))
    assert "top_p" not in params


def test_build_chat_request_params_reasoning_model_top_p_default_does_not_raise():
    """top_p=0.95 (the default) must NOT raise a ValueError for reasoning models."""
    model = _make_openai_model("o3-mini")
    # Should not raise
    model._build_chat_request_params(**_standard_chat_params(top_p=0.95))


def test_build_chat_request_params_reasoning_model_top_p_custom_raises():
    """A non-default, non-None top_p must raise ValueError for reasoning models."""
    model = _make_openai_model("o1")
    with pytest.raises(ValueError, match="top_p"):
        model._build_chat_request_params(**_standard_chat_params(top_p=0.5))


def test_build_chat_request_params_gpt5_reasoning_top_p_none_does_not_raise():
    """gpt-5 is treated as a reasoning model; top_p=None should not raise."""
    model = _make_openai_model("gpt-5")
    model._build_chat_request_params(**_standard_chat_params(top_p=None))


def test_build_chat_request_params_gpt5_reasoning_top_p_custom_raises():
    """gpt-5 is treated as a reasoning model; a non-default top_p should raise ValueError."""
    model = _make_openai_model("gpt-5")
    with pytest.raises(ValueError, match="top_p"):
        model._build_chat_request_params(**_standard_chat_params(top_p=0.7))


def test_build_chat_request_params_reasoning_model_top_p_zero_raises():
    """top_p=0.0 is not 0.95 and not None, so it should raise for reasoning models."""
    model = _make_openai_model("o1")
    with pytest.raises(ValueError, match="top_p"):
        model._build_chat_request_params(**_standard_chat_params(top_p=0.0))


# ---------------------------------------------------------------------------
# Tests: InferenceConfig accepts top_p as Optional[float]
# ---------------------------------------------------------------------------


def test_inference_config_top_p_accepts_none():
    """InferenceConfig must accept top_p=None (float | None type annotation)."""
    from nemo_skills.inference.generate import InferenceConfig

    cfg = InferenceConfig(top_p=None)
    assert cfg.top_p is None


def test_inference_config_top_p_default_is_0_95():
    """InferenceConfig.top_p should default to 0.95."""
    from nemo_skills.inference.generate import InferenceConfig

    cfg = InferenceConfig()
    assert cfg.top_p == 0.95


def test_inference_config_top_p_accepts_custom_float():
    """InferenceConfig must accept an arbitrary float for top_p."""
    from nemo_skills.inference.generate import InferenceConfig

    cfg = InferenceConfig(top_p=0.8)
    assert cfg.top_p == 0.8