# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAFIM evaluation via the ``safim`` package (install-on-the-fly), LiveCodeBench-style.

Follows the same flow as ``livecodebench.py``: preprocess JSONL in place, install
``safim`` from git if needed, run ``safim.evaluate.evaluate`` (uses ExecEval at
``http://localhost:{port}`` inside the environment), merge ``graded_list`` into
rows, rename harness output to ``*_eval_results-saved.json``.

**Data expectations (JSONL after generation)**

- ``prefix``, ``suffix``, ``language``, ``task_id``, ``generation`` / ``completion``.
- The upstream ``safim.evaluate`` API loads completions by ``task_id``; this file
  must contain **at most one row per ``task_id``** (same constraint as the
  reference ``safim/evaluate.py`` dict merge).  Use separate seed files for pass@k.

**Config**

- ``subset``: HuggingFace subset name passed as ``completion_type`` (e.g.
  ``api``, ``block``, ``control``).
- ``execeval_port``: ExecEval port (default ``5000``); the ``safim`` client uses
  ``localhost`` only.
- ``sandbox``: **Required** — NeMo code sandbox (shell). Evaluation does not run
  on the driver machine if the sandbox is unreachable.

See also: https://github.com/wasiahmad/safim (library usage in README).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
import textwrap
from dataclasses import field, fields
from pathlib import Path
from typing import Any

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.evaluation.evaluator.livecodebench import (
    execute_in_sandbox_with_retries,
    is_sandbox_available,
    sandbox_context,
)
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

# Default matches SAFIM README (execution-based fork).
SAFIM_GIT_URL = "git+https://github.com/wasiahmad/safim.git"


def _remove_overlap(preceding_text: str, following_text: str, *, truncate_from: str = "following") -> str:
    """Trim overlap between infill and prefix/suffix (same idea as human-eval-infilling)."""
    assert truncate_from in ("preceding", "following")
    preceding_len = len(preceding_text)
    following_len = len(following_text)
    for i in range(min(preceding_len, following_len), 0, -1):
        if truncate_from == "following":
            overlap = preceding_text[-i:]
            if overlap.strip() == "" and "\n" not in overlap:
                continue
            if following_text.startswith(overlap):
                return following_text[i:]
        else:
            overlap = following_text[:i]
            if overlap.strip() == "" and "\n" not in overlap:
                continue
            if preceding_text.endswith(overlap):
                return preceding_text[:-i]
    return following_text if truncate_from == "following" else preceding_text


def _fim_postprocess(sample: dict) -> dict:
    sample["completion"] = _remove_overlap(sample["prefix"], sample["completion"], truncate_from="following")
    sample["completion"] = _remove_overlap(sample["completion"], sample["suffix"], truncate_from="preceding")
    return sample


def _fence_language(dataset_language: str) -> str:
    lang = str(dataset_language).lower()
    if lang in ("cpp", "c++"):
        return "cpp"
    if lang in ("csharp", "c#"):
        return "csharp"
    return lang


@nested_dataclass(kw_only=True)
class SafimEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for SAFIM via ``safim.evaluate`` (LiveCodeBench-style)."""

    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    interpreter: str = "python"
    # HuggingFace subset: api | block | control (passed to ``safim.evaluate``).
    subset: str = "api"
    # Optional filter; ``None`` evaluates all languages in the subset.
    language: str | None = None
    execeval_port: int = 5000
    num_retries: int = 3
    eval_timeout_buffer_s: float = 120.0
    fim_postprocess: bool = True
    safim_git_url: str = SAFIM_GIT_URL


def _safim_cfg_subset(cfg: dict) -> dict:
    allowed = {f.name for f in fields(SafimEvaluatorConfig)}
    return {k: v for k, v in cfg.items() if k in allowed}


def _preprocess_safim_jsonl(jsonl_file: str, fim_postprocess: bool) -> list[dict]:
    """Read JSONL, fence-extract completion + FIM trim, write back; return samples."""
    path = Path(jsonl_file)
    with open(path, encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh]

    if not raw:
        raise ValueError(f"No samples found in {jsonl_file}")

    seen: set[str] = set()
    duplicates: list[str] = []
    prepared: list[dict] = []
    for idx, row in enumerate(raw):
        row = dict(row)
        fence_lang = _fence_language(row["language"])
        row = preprocess_code(row, language=fence_lang, strip_whitespace=False)
        if fim_postprocess:
            row = _fim_postprocess(row)
        tid = str(row.get("task_id", f"row-{idx}"))
        row["task_id"] = tid
        if tid in seen:
            duplicates.append(tid)
        seen.add(tid)
        prepared.append(row)

    if duplicates:
        raise ValueError(
            "safim.evaluate() keeps one completion per task_id (dict merge over JSONL). "
            f"Duplicate task_id(s) in {jsonl_file}: {sorted(set(duplicates))}. "
            "Use one row per task per file (e.g. separate output-rs*.jsonl seeds)."
        )

    with open(path, "w", encoding="utf-8") as fh:
        for row in prepared:
            fh.write(json.dumps(row) + "\n")

    return prepared


def _postprocess_safim_results(jsonl_file: str, samples: list[dict]) -> None:
    """Merge ``safim`` harness output into JSONL and rename to ``*_eval_results-saved.json``."""
    results_file = jsonl_file[:-6] + "_eval_results.json"
    with open(results_file, encoding="utf-8") as fh:
        eval_payload = json.load(fh)

    eval_block = eval_payload.get("eval") or {}

    with open(jsonl_file, "w", encoding="utf-8") as fh:
        for s in samples:
            tid = str(s["task_id"])
            entry = eval_block.get(tid)
            passed = False
            result = "MISSING"
            if isinstance(entry, list) and entry:
                passed = bool(entry[0].get("passed", False))
                result = str(entry[0].get("result", ""))
            s["graded_list"] = [passed]
            s["safim_eval_result"] = result
            s["execeval_passed"] = passed
            fh.write(json.dumps(s) + "\n")

    shutil.move(results_file, jsonl_file[:-6] + "_eval_results-saved.json")
    LOG.info("Finished processing %s, results saved.", jsonl_file)


async def _install_safim_in_sandbox(sandbox, eval_config: SafimEvaluatorConfig) -> bool:
    pip_cmd = "pip" if eval_config.interpreter == "python" else "pypy3 -m pip"
    cmd = f"{pip_cmd} install {eval_config.safim_git_url}"
    out, _ = await execute_in_sandbox_with_retries(
        sandbox, eval_config.num_retries, cmd, language="shell", timeout=300
    )
    if out.get("process_status") != "completed":
        LOG.warning("Failed to install safim: %s", out.get("stderr", "Unknown error"))
        return False
    LOG.info("Successfully installed safim.")
    return True


async def eval_safim_async(eval_config: SafimEvaluatorConfig) -> None:
    async with sandbox_context(eval_config.sandbox) as sandbox:
        if not await _install_safim_in_sandbox(sandbox, eval_config):
            return

        jsonl_file = str(Path(eval_config.input_file).resolve())
        try:
            samples = _preprocess_safim_jsonl(jsonl_file, eval_config.fim_postprocess)
        except ValueError as e:
            LOG.error("%s", e)
            return

        output_json = jsonl_file[:-6] + "_eval_results.json"
        lang_repr = "None" if eval_config.language is None else repr(eval_config.language)
        eval_code = textwrap.dedent(
            f"""
            from safim.evaluate import evaluate
            evaluate(
                {repr(eval_config.subset)},
                {repr(jsonl_file)},
                {repr(output_json)},
                language={lang_repr},
                port={int(eval_config.execeval_port)},
            )
            """
        )

        cmd = f"{eval_config.interpreter} -c {shlex.quote(eval_code)}"
        timeout_s = max(300.0, len(samples) * 30.0 + eval_config.eval_timeout_buffer_s)
        output, _ = await execute_in_sandbox_with_retries(
            sandbox,
            eval_config.num_retries,
            cmd,
            language="shell",
            timeout=timeout_s,
            max_output_characters=100_000,
        )
        if output.get("process_status") != "completed":
            LOG.error("SAFIM evaluation failed for %s. Stderr: %s", jsonl_file, output.get("stderr"))
            return

        results_path = Path(output_json)
        if not results_path.is_file():
            LOG.error("Expected SAFIM output at %s but file is missing.", output_json)
            return

        _postprocess_safim_results(jsonl_file, samples)


def eval_safim(cfg: dict) -> dict[str, Any]:
    """Install ``safim`` if needed and run ``safim.evaluate.evaluate`` (LiveCodeBench-style)."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    if not isinstance(cfg, dict):
        raise TypeError(f"eval_safim expected dict-like config, got {type(cfg)}")

    eval_cfg = SafimEvaluatorConfig(_init_nested=True, **_safim_cfg_subset(cfg))

    if not eval_cfg.input_file:
        raise ValueError("eval_safim requires input_file in eval config.")

    if eval_cfg.interpreter not in ("python", "pypy3"):
        raise ValueError("SAFIM interpreter must be 'python' or 'pypy3'.")

    sandbox_is_ready = asyncio.run(is_sandbox_available(eval_cfg.sandbox))
    if not sandbox_is_ready:
        raise RuntimeError(
            "SAFIM evaluation requires a reachable NeMo code sandbox. "
            "Start the sandbox service and ensure eval_config.sandbox (host, port, ssh_server, …) is correct."
        )

    asyncio.run(eval_safim_async(eval_cfg))

    return {}
