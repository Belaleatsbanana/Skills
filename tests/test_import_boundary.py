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

"""Tests that enforce the Core/Pipeline import boundary.

The architectural rule is:
  - Pipeline can import from Core
  - Core CANNOT import from Pipeline (at the top level)

Core modules are everything under nemo_skills/ EXCEPT nemo_skills/pipeline/.
This test statically parses Python files to detect violations.
"""

import ast
from pathlib import Path

# Root of the nemo_skills package
NEMO_SKILLS_ROOT = Path(__file__).parent.parent / "nemo_skills"

# Modules classified as "core" (everything except pipeline)
CORE_DIRS = [
    NEMO_SKILLS_ROOT / "inference",
    NEMO_SKILLS_ROOT / "evaluation",
    NEMO_SKILLS_ROOT / "dataset",
    NEMO_SKILLS_ROOT / "prompt",
    NEMO_SKILLS_ROOT / "mcp",
    NEMO_SKILLS_ROOT / "code_execution",
    NEMO_SKILLS_ROOT / "conversion",
    NEMO_SKILLS_ROOT / "training",
]

CORE_FILES = [
    NEMO_SKILLS_ROOT / "utils.py",
    NEMO_SKILLS_ROOT / "file_utils.py",
]

# Import prefixes that core must NOT use at the top level
FORBIDDEN_PREFIXES = [
    "nemo_skills.pipeline",
]


def _get_all_core_python_files():
    """Collect all .py files in core modules."""
    files = []
    for core_dir in CORE_DIRS:
        if core_dir.is_dir():
            files.extend(core_dir.rglob("*.py"))
    for core_file in CORE_FILES:
        if core_file.is_file():
            files.append(core_file)
    return sorted(files)


def _get_top_level_imports(filepath):
    """Parse a Python file and return all top-level import statements.

    Returns a list of (line_number, module_name) tuples for imports
    that are NOT inside function/method bodies (i.e., they are at
    module scope or inside class bodies at module scope).
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))
        elif isinstance(node, ast.ClassDef):
            # Also check class-level imports (not inside methods)
            for class_node in ast.iter_child_nodes(node):
                if isinstance(class_node, ast.Import):
                    for alias in class_node.names:
                        imports.append((class_node.lineno, alias.name))
                elif isinstance(class_node, ast.ImportFrom):
                    if class_node.module:
                        imports.append((class_node.lineno, class_node.module))
    return imports


def test_core_does_not_import_pipeline_at_top_level():
    """Ensure core modules don't have top-level imports from pipeline.

    Lazy imports inside functions (for deprecation redirects) are allowed.
    Only top-level (module scope or class scope) imports are checked.
    """
    violations = []

    for py_file in _get_all_core_python_files():
        for lineno, module_name in _get_top_level_imports(py_file):
            for prefix in FORBIDDEN_PREFIXES:
                if module_name.startswith(prefix) or module_name == prefix:
                    rel_path = py_file.relative_to(NEMO_SKILLS_ROOT.parent)
                    violations.append(f"  {rel_path}:{lineno} imports '{module_name}'")

    assert not violations, (
        "Core modules must not have top-level imports from pipeline.\n"
        "Found violations:\n" + "\n".join(violations) + "\n\n"
        "If this is a deprecation redirect, move the import inside the function body.\n"
        "If you need pipeline functionality in core, reconsider the architecture."
    )


def test_core_does_not_import_nemo_run_at_top_level():
    """Ensure core modules don't directly import nemo_run at the top level."""
    violations = []

    for py_file in _get_all_core_python_files():
        for lineno, module_name in _get_top_level_imports(py_file):
            if module_name.startswith("nemo_run") or module_name == "nemo_run":
                rel_path = py_file.relative_to(NEMO_SKILLS_ROOT.parent)
                violations.append(f"  {rel_path}:{lineno} imports '{module_name}'")

    assert not violations, (
        "Core modules must not have top-level imports of nemo_run.\n"
        "Found violations:\n" + "\n".join(violations) + "\n\n"
        "nemo_run is a pipeline dependency. Use it only in nemo_skills/pipeline/."
    )
