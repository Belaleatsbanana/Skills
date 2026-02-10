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

"""Tests that verify dependency isolation between core, pipeline, and benchmark extras.

Each test creates a fresh virtualenv, installs specific extras, and checks that
only the expected modules are importable. These tests are slow (~30s each) due
to venv creation and pip install, so they're marked with @pytest.mark.isolation.

Run with:
    pytest tests/test_dependency_isolation.py -v
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _make_venv(tmp_path, extras=None):
    """Create a venv and install nemo-skills with the given extras."""
    venv_dir = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--python", "3.10", str(venv_dir)], check=True, capture_output=True)
    python = str(venv_dir / "bin" / "python")

    if extras:
        install_target = f"{REPO_ROOT}[{extras}]"
    else:
        install_target = str(REPO_ROOT)

    subprocess.run(
        ["uv", "pip", "install", "--python", python, install_target],
        check=True,
        capture_output=True,
    )
    return python


def _can_import(python, module):
    """Return True if the module imports successfully in the given python."""
    result = subprocess.run(
        [python, "-c", f"import {module}"],
        capture_output=True,
    )
    return result.returncode == 0


def _import_check(python, statement):
    """Return True if an arbitrary import statement succeeds."""
    result = subprocess.run(
        [python, "-c", statement],
        capture_output=True,
    )
    return result.returncode == 0


def test_core_only(tmp_path):
    """Base install (no extras) should have core modules but NOT pipeline or nemo_run."""
    python = _make_venv(tmp_path)

    # Core modules must work
    assert _import_check(python, "from nemo_skills.dataset.utils import get_dataset_module")
    assert _import_check(python, "from nemo_skills.inference.generate import GenerationTask")
    assert _import_check(python, "from nemo_skills.evaluation.evaluator import evaluate, EVALUATOR_MAP")

    # Pipeline must NOT be importable (missing nemo_run)
    assert not _can_import(python, "nemo_run"), "nemo_run should not be installed with core-only"
    assert not _import_check(python, "from nemo_skills.pipeline.dataset import get_dataset_module"), (
        "pipeline.dataset should not import without pipeline deps"
    )


def test_pipeline_extra(tmp_path):
    """.[pipeline] should have core + pipeline modules, nemo_run available."""
    python = _make_venv(tmp_path, extras="pipeline")

    # Core
    assert _import_check(python, "from nemo_skills.dataset.utils import get_dataset_module")
    assert _import_check(python, "from nemo_skills.inference.generate import GenerationTask")

    # Pipeline
    assert _can_import(python, "nemo_run")
    assert _import_check(python, "from nemo_skills.pipeline.dataset import get_dataset_module")


def test_all_extra(tmp_path):
    """.[all] should have everything including benchmark deps."""
    python = _make_venv(tmp_path, extras="all")

    # Core
    assert _import_check(python, "from nemo_skills.dataset.utils import get_dataset_module")

    # Pipeline
    assert _can_import(python, "nemo_run")
    assert _import_check(python, "from nemo_skills.pipeline.dataset import get_dataset_module")

    # Benchmark deps
    assert _can_import(python, "faiss")
    assert _can_import(python, "sacrebleu")
