# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""SAFIM evaluation via ExecEval HTTP API and pass@k (draft).

This module mirrors the contract of ``ExecEval/eval_scripts/eval_passk.py``:
samples must expose ``source_code``, ``lang`` (ExecEval *runtime_name*),
``src_uid``, and ``task_id``.  See ExecEval README for ``unittest_db`` layout.

**Data expectations (JSONL after generation)**

- ``prefix``, ``suffix``, ``language`` (dataset language slug, e.g. ``python``).
- ``generation``: model output (fenced code extracted like other code evals).
- ``src_uid``: key into ``unittest_file`` (must be present in the benchmark;
  extend ``prepare.py`` if the Hub row id is under another field).
- ``task_id``: logical task id for pass@k grouping (same id across *n*
  samples for one problem).

**Infra**

- ExecEval server reachable at ``execeval_url`` (e.g. service inside sandbox
  with port forwarding, or ``http://127.0.0.1:5000``).
- ``unittest_file``: JSON object mapping ``src_uid`` -> list of unit tests
  (``input`` / ``output`` fields per ExecEval).
- Optional: run ``eval_passk.py`` manually inside the sandbox instead of this
  client; this implementation inlines the HTTP path using ``httpx``.

TODO (review pass):

- Align ``language`` -> ExecEval runtime names with your sandbox image.
- Persist SAFIM-native columns in ``prepare.py`` (``src_uid``, tests export).
- Sandbox-only orchestration (single shell job calling ``eval_passk.py``) if
  the driver cannot reach ExecEval HTTP.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import field, fields
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

PASSED_OUTCOME = "PASSED"

# Dataset ``language`` (lowercase slug) -> ExecEval ``language`` / runtime_name
DEFAULT_LANGUAGE_TO_RUNTIME: dict[str, str] = {
    "python": "Python 3",
    "cpp": "GNU C++17",
    "java": "Java 11",
    "javascript": "Node.js",
    "csharp": ".NET Core C#",
}


def estimate_pass_at_k(
    num_samples: int | list[int] | np.ndarray,
    num_correct: list[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    """Unbiased pass@k estimator (same as ExecEval ``eval_passk.py``)."""

    def estimator(n: int, c: int, k_val: int) -> float:
        if n - c < k_val:
            return 1.0
        return 1.0 - float(np.prod(1.0 - k_val / np.arange(n - c + 1, n + 1)))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


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


def _resolve_runtime(dataset_language: str, overrides: dict[str, str]) -> str | None:
    slug = str(dataset_language).lower()
    if slug in ("c++",):
        slug = "cpp"
    if slug in ("c#",):
        slug = "csharp"
    if slug in overrides:
        return overrides[slug]
    return DEFAULT_LANGUAGE_TO_RUNTIME.get(slug)


def _get_execeval_runtimes(client: httpx.Client, base_url: str) -> set[str]:
    url = f"{base_url.rstrip('/')}/api/all_runtimes"
    response = client.get(url)
    response.raise_for_status()
    data = response.json()
    return {r["runtime_name"] for r in data}


def _execute_sample(
    client: httpx.Client,
    base_url: str,
    *,
    runtime: str,
    source_code: str,
    unittests: list[dict],
    limits: dict[str, Any],
    block_network: bool,
    stop_on_first_fail: bool,
    use_sanitizer: bool,
) -> list[dict] | dict:
    url = f"{base_url.rstrip('/')}/api/execute_code"
    payload = {
        "language": runtime,
        "source_code": source_code,
        "unittests": unittests,
        "limits": limits,
        "block_network": block_network,
        "stop_on_first_fail": stop_on_first_fail,
        "use_sanitizer": use_sanitizer,
    }
    response = client.post(url, json=payload)
    response.raise_for_status()
    body = response.json()
    if "data" not in body:
        return body
    return body["data"]


def _all_unittests_passed(results: list[dict]) -> bool:
    return all(u.get("exec_outcome") == PASSED_OUTCOME for u in results)


@nested_dataclass(kw_only=True)
class SafimEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for SAFIM + ExecEval pass@k (draft)."""

    # ExecEval HTTP API (server must be running, e.g. inside sandbox).
    execeval_url: str = "http://127.0.0.1:5000"
    # Used to locate default ``limits_by_lang.yaml`` if ``limits_by_lang_file`` is unset.
    exec_eval_root: str = "/home/wahmad/Desktop/workspace/ExecEval"
    # JSON mapping src_uid -> list[ExtendedUnittest dict]
    unittest_file: str | None = None
    limits_by_lang_file: str | None = None
    k: str = "1,10"
    n_workers: int = 32
    block_network: bool = True
    stop_on_first_fail: bool = True
    use_sanitizer: bool = False
    request_timeout_s: float = 120.0
    fim_postprocess: bool = True
    # Merges onto DEFAULT_LANGUAGE_TO_RUNTIME (keys: dataset language slug).
    language_to_runtime: dict = field(default_factory=dict)
    # Write pass@k aggregates next to the evaluated JSONL.
    write_pass_at_k_json: bool = True
    # Reserved for future sandbox-wrapped invocations (ignored by HTTP client path).
    sandbox: dict = field(default_factory=dict)


def _safim_cfg_subset(cfg: dict) -> dict:
    allowed = {f.name for f in fields(SafimEvaluatorConfig)}
    return {k: v for k, v in cfg.items() if k in allowed}


def eval_safim(cfg: dict) -> dict[str, float]:
    """Batch-evaluate ``input_file`` with ExecEval; merge outcomes into rows; return pass@k."""
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
    if not eval_cfg.unittest_file:
        raise ValueError(
            "eval_safim requires unittest_file (ExecEval unittest_db JSON). "
            "Point it at a SAFIM-derived unittest export."
        )

    unittest_path = Path(eval_cfg.unittest_file).expanduser()
    if not unittest_path.is_file():
        raise FileNotFoundError(f"unittest_file not found: {unittest_path}")

    limits_path = eval_cfg.limits_by_lang_file
    if limits_path is None:
        limits_path = str(Path(eval_cfg.exec_eval_root).expanduser() / "eval_scripts" / "limits_by_lang.yaml")
    limits_path = str(Path(limits_path).expanduser())
    with open(limits_path, encoding="utf-8") as fh:
        limits_by_lang = yaml.safe_load(fh)

    with open(unittest_path, encoding="utf-8") as fh:
        unittest_db: dict[str, list] = json.load(fh)

    jsonl_path = Path(eval_cfg.input_file)
    with open(jsonl_path, encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh]

    ks = [int(x.strip()) for x in eval_cfg.k.split(",") if x.strip()]
    lang_overrides = dict(eval_cfg.language_to_runtime) if eval_cfg.language_to_runtime else {}

    prepared: list[dict] = []
    for row in samples:
        row = dict(row)
        fence_lang = _fence_language(row["language"])
        row = preprocess_code(row, language=fence_lang, strip_whitespace=False)
        if eval_cfg.fim_postprocess:
            row = _fim_postprocess(row)
        source_code = f"{row['prefix']}{row.get('completion', '')}{row['suffix']}"
        row["_safim_source_code"] = source_code
        prepared.append(row)

    base_url = eval_cfg.execeval_url.rstrip("/")
    timeout = httpx.Timeout(eval_cfg.request_timeout_s)

    with httpx.Client(timeout=timeout) as client:
        supported = _get_execeval_runtimes(client, base_url)
        LOG.info("ExecEval runtimes available: %d (%s ...)", len(supported), next(iter(supported), ""))

        tasks: list[tuple[int, str, str, str, list]] = []
        for idx, row in enumerate(prepared):
            src_uid = str(row.get("src_uid", ""))
            task_id = str(row.get("task_id", src_uid or f"row-{idx}"))
            row["task_id"] = task_id

            runtime = _resolve_runtime(row["language"], lang_overrides)
            if not runtime or runtime not in supported:
                LOG.warning(
                    "Skipping idx=%s: runtime %r not supported by ExecEval (language=%r).",
                    idx,
                    runtime,
                    row.get("language"),
                )
                continue
            if not src_uid or src_uid not in unittest_db:
                LOG.warning("Skipping idx=%s: missing unittest entry for src_uid=%r.", idx, src_uid)
                continue
            unittests = unittest_db[src_uid]
            if not unittests:
                LOG.warning("Skipping idx=%s: empty unittest list for src_uid=%r.", idx, src_uid)
                continue
            limits = limits_by_lang.get(runtime)
            if limits is None:
                LOG.warning("No resource limits for runtime %r; using ExecEval defaults on server.", runtime)
            tasks.append((idx, task_id, runtime, row["_safim_source_code"], unittests))

        results_by_task: dict[str, list[tuple[int, list[dict] | dict]]] = defaultdict(list)

        def run_one(args: tuple[int, str, str, str, list]):
            idx, task_id, runtime, source_code, uts = args
            with httpx.Client(timeout=timeout) as thread_client:
                try:
                    out = _execute_sample(
                        thread_client,
                        base_url,
                        runtime=runtime,
                        source_code=source_code,
                        unittests=uts,
                        limits=limits_by_lang.get(runtime, {}),
                        block_network=eval_cfg.block_network,
                        stop_on_first_fail=eval_cfg.stop_on_first_fail,
                        use_sanitizer=eval_cfg.use_sanitizer,
                    )
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    return idx, task_id, {"error": str(exc)}
            return idx, task_id, out

        LOG.info("Submitting %d ExecEval jobs with %d workers...", len(tasks), eval_cfg.n_workers)
        with ThreadPoolExecutor(max_workers=eval_cfg.n_workers) as pool:
            futures = [pool.submit(run_one, t) for t in tasks]
            for fut in as_completed(futures):
                idx, task_id, out = fut.result()
                results_by_task[task_id].append((idx, out))

    # Attach per-row results (sort by idx for stability)
    idx_to_row = {i: r for i, r in enumerate(prepared)}
    for row in prepared:
        row.setdefault("execeval_passed", False)

    for task_id, lst in results_by_task.items():
        lst.sort(key=lambda x: x[0])
        for idx, out in lst:
            row = idx_to_row[idx]
            if isinstance(out, dict) and "error" in out:
                row["execeval_error"] = out["error"]
                row["execeval_passed"] = False
                row["execeval_unittests"] = []
                continue
            if not isinstance(out, list):
                row["execeval_error"] = f"unexpected response: {out!r}"
                row["execeval_passed"] = False
                row["execeval_unittests"] = []
                continue
            row["execeval_unittests"] = out
            row["execeval_passed"] = _all_unittests_passed(out)

    # Drop staging keys
    for row in prepared:
        row.pop("_safim_source_code", None)

    pass_at_k: dict[str, float] = {}
    totals: list[int] = []
    correct: list[int] = []
    for _task_id, lst in sorted(results_by_task.items(), key=lambda x: x[0]):
        lst.sort(key=lambda x: x[0])
        passed_flags: list[bool] = []
        for _idx, out in lst:
            if isinstance(out, dict) and "error" in out:
                passed_flags.append(False)
            elif isinstance(out, list):
                passed_flags.append(_all_unittests_passed(out))
            else:
                passed_flags.append(False)
        totals.append(len(passed_flags))
        correct.append(sum(passed_flags))

    if totals:
        total_arr = np.array(totals)
        correct_arr = np.array(correct)
        for k in ks:
            if (total_arr >= k).all():
                key = f"pass@{k}"
                pass_at_k[key] = float(estimate_pass_at_k(total_arr, correct_arr, k).mean())
            else:
                LOG.warning(
                    "Skipping %s: some tasks have fewer than %d samples (need equal n per task for this estimator).",
                    f"pass@{k}",
                    k,
                )

    LOG.info("pass@k (ExecEval): %s", pass_at_k)

    if eval_cfg.write_pass_at_k_json and pass_at_k:
        sidecar = jsonl_path.with_name(jsonl_path.stem + "_exec_eval_pass_at_k.json")
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(pass_at_k, fh, indent=2)
        LOG.info("Wrote %s", sidecar)

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in prepared:
            fh.write(json.dumps(row) + "\n")

    return pass_at_k
