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

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

# Shared go-judge request fragments (do not mutate).
_GO_JUDGE_ENV = ["PATH=/usr/bin:/bin:/usr/local/bin"]
_LANG_ALIAS_MAP = {
    "py": "python",
    "python3": "python",
    "c++": "cpp",
    "cxx": "cpp",
}


def seconds_to_go_judge_duration_ns(seconds: float) -> int:
    return max(1, int(float(seconds) * 1_000_000_000))


def _eval_results_json_path(jsonl_path: str) -> str:
    if jsonl_path.endswith(".jsonl"):
        return jsonl_path[:-6] + "_eval_results.json"
    return jsonl_path + "_eval_results.json"


@dataclass
class GoJudgeClient:
    """HTTP client for go-judge ``POST /run`` and ``DELETE /file/:id``."""

    host: str = "127.0.0.1"
    port: int = 5050
    auth_token: str | None = None
    http_timeout: float = 120.0
    compile_memory_bytes: int = 536_870_912
    run_memory_bytes: int = 8 * (1024**3)
    proc_limit: int = 100
    compile_stdout_max: int = 262_144
    compile_stderr_max: int = 262_144
    run_stdout_max: int = 67_108_864
    run_stderr_max: int = 65_536
    gxx_path: str = "/usr/bin/g++"
    python_path: str = "/usr/bin/python3"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def close(self) -> None:
        pass

    def run_cmd(self, payload: dict) -> list[dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/run",
            data=data,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"go-judge /run HTTP {resp.status}")
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"go-judge /run HTTP {e.code}: {err_body}") from e
        parsed = json.loads(body)
        if not isinstance(parsed, list):
            raise RuntimeError(f"go-judge /run expected JSON list, got {type(parsed)}")
        return parsed

    def delete_file(self, file_id: str) -> None:
        if not file_id:
            return
        req = urllib.request.Request(
            f"{self.base_url}/file/{urllib.parse.quote(file_id, safe='')}",
            method="DELETE",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=min(self.http_timeout, 30.0)) as resp:
                if resp.status not in (200, 204):
                    raise RuntimeError(f"go-judge DELETE /file HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            if e.code in (404,):
                return
            raise

    @classmethod
    def from_config(cls, cfg: dict) -> GoJudgeClient:
        return cls(
            host=str(cfg.get("host") or os.environ.get("NEMO_SKILLS_GO_JUDGE_HOST", "127.0.0.1")),
            port=int(cfg.get("port") or os.environ.get("NEMO_SKILLS_GO_JUDGE_PORT", "5050")),
            auth_token=(cfg.get("auth_token") or os.environ.get("NEMO_SKILLS_GO_JUDGE_TOKEN") or None),
            http_timeout=float(cfg.get("http_timeout", 120.0)),
            compile_memory_bytes=int(cfg.get("compile_memory_bytes", 536_870_912)),
            run_memory_bytes=int(cfg.get("run_memory_bytes", 8 * (1024**3))),
            proc_limit=int(cfg.get("proc_limit", 100)),
            compile_stdout_max=int(cfg.get("compile_stdout_max", 262_144)),
            compile_stderr_max=int(cfg.get("compile_stderr_max", 262_144)),
            run_stdout_max=int(cfg.get("run_stdout_max", 67_108_864)),
            run_stderr_max=int(cfg.get("run_stderr_max", 65_536)),
            gxx_path=str(cfg.get("gxx_path", "/usr/bin/g++")),
            python_path=str(cfg.get("python_path", "/usr/bin/python3")),
        )


@dataclass
class StdioTestCase:
    input: str
    output: str


def index_zip_files(directory_path: Path) -> dict[str, Path]:
    target_dir = Path(directory_path)
    if not target_dir.exists():
        LOG.warning("Test directory not found: %s", target_dir)
        return {}
    return {p.stem: p for p in target_dir.glob("*.zip")}


def read_stdio_tests_from_zip(zip_path: Path) -> list[StdioTestCase]:
    """Match LiveCodeBench-Pro layout: testdata/*.in with sibling .ans."""
    loaded: list[StdioTestCase] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            in_files = sorted(f for f in names if f.startswith("testdata/") and f.endswith(".in"))
            for in_file in in_files:
                ans_file = in_file[:-3] + ".ans"
                if ans_file not in names:
                    continue
                with zf.open(in_file) as f:
                    in_txt = f.read().decode("utf-8", errors="replace")
                with zf.open(ans_file) as f:
                    ans_txt = f.read().decode("utf-8", errors="replace")
                loaded.append(StdioTestCase(input=in_txt, output=ans_txt))
    except Exception as e:
        LOG.warning("Error reading zip %s: %s", zip_path, e)
    return loaded


def _truncatefn(s: str, length: int = 300) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def _match_outputs(prediction: Any, gt_out: Any) -> bool:
    if prediction == gt_out:
        return True
    try:
        if json.dumps(prediction) == json.dumps(gt_out):
            return True
    except (TypeError, ValueError):
        pass
    try:
        if np.allclose(float(prediction), float(gt_out)):
            return True
    except (TypeError, ValueError, OverflowError):
        pass
    if isinstance(prediction, list):
        try:
            if len(prediction) == len(gt_out):
                for i in range(len(prediction)):
                    if not np.allclose(float(prediction[i]), float(gt_out[i])):
                        return False
                return True
        except (TypeError, ValueError, OverflowError):
            pass
    return False


def _get_stripped_lines(val: str) -> list[str]:
    val = val.strip()
    return [ln.strip() for ln in val.split("\n")]


def _stdio_line_matches(pred_line: str, gt_line: str) -> bool:
    if pred_line == gt_line:
        return True
    try:
        pred_v = json.loads(pred_line)
        gt_v = json.loads(gt_line)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return _match_outputs(pred_v, gt_v)


def _convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def compare_stdio_to_expected(prediction: str, expected: str) -> tuple[bool, dict | None]:
    """Line-wise stdout comparison (aligned with LiveCodeBench ``testing_util``)."""
    plines = _get_stripped_lines(prediction)
    glines = _get_stripped_lines(expected)
    if len(plines) != len(glines):
        return False, {
            "output": _truncatefn(prediction),
            "expected": _truncatefn(expected),
            "error_code": -2,
            "error_message": "Wrong answer: mismatched output length",
        }
    wa_base: dict | None = None

    def _wa() -> dict:
        nonlocal wa_base
        if wa_base is None:
            wa_base = {
                "output": _truncatefn(prediction),
                "expected": _truncatefn(expected),
                "error_code": -2,
            }
        return wa_base

    for idx, (ps, gs) in enumerate(zip(plines, glines)):
        if _stdio_line_matches(ps, gs):
            continue
        meta = _wa()
        meta["error_message"] = f"Wrong answer at output_line_idx={idx}: {_truncatefn(ps)} != {_truncatefn(gs)}"
        ok_p, dec_p = _convert_line_to_decimals(ps)
        if not ok_p:
            return False, meta
        ok_g, dec_g = _convert_line_to_decimals(gs)
        if not ok_g:
            return False, meta
        if dec_p == dec_g:
            continue
        try:
            if len(dec_p) == len(dec_g):
                pred_f = [float(d) for d in dec_p]
                gt_f = [float(d) for d in dec_g]
                if pred_f == gt_f or np.allclose(pred_f, gt_f):
                    continue
        except (TypeError, ValueError, OverflowError):
            pass
        return False, meta
    return True, None


def _normalize_language(lang: str) -> str:
    k = (lang or "cpp").strip().lower()
    return _LANG_ALIAS_MAP.get(k, k)


def _run_one_go_judge_case(
    client: GoJudgeClient,
    stdin: str,
    time_limit_s: float,
    memory_bytes: int,
    copy_in: dict,
    run_args: list[str],
) -> tuple[str, dict]:
    ns = seconds_to_go_judge_duration_ns(time_limit_s)
    payload = {
        "cmd": [
            {
                "args": run_args,
                "env": _GO_JUDGE_ENV,
                "files": [
                    {"content": stdin},
                    {"name": "stdout", "max": client.run_stdout_max},
                    {"name": "stderr", "max": client.run_stderr_max},
                ],
                "cpuLimit": ns,
                "clockLimit": ns,
                "memoryLimit": memory_bytes,
                "procLimit": client.proc_limit,
                "copyIn": copy_in,
            }
        ]
    }
    results = client.run_cmd(payload)
    r0 = results[0]
    files = r0.get("files") or {}
    return (files.get("stdout") or ""), r0


def _go_judge_exit_ok(exit_status: Any) -> bool:
    """True if process exit code is zero (go-judge may encode as int, str, or float in JSON)."""
    if exit_status is None:
        return False
    try:
        return int(exit_status) == 0
    except (TypeError, ValueError):
        try:
            return float(exit_status) == 0.0
        except (TypeError, ValueError):
            return False


def _go_judge_run_ok(r0: dict) -> bool:
    return r0.get("status") == "Accepted" and _go_judge_exit_ok(r0.get("exitStatus"))


def _go_judge_result_meta(r0: dict) -> dict[str, Any]:
    """Subset of go-judge ``Result`` for logging/metadata (see go-judge README *Return Status*)."""
    meta: dict[str, Any] = {
        "status": r0.get("status"),
        "exitStatus": r0.get("exitStatus"),
    }
    err = r0.get("error")
    if err:
        meta["error"] = err
    return meta


def _summarize_return_status(meta: dict[str, Any]) -> str:
    """One-line summary of go-judge *Return Status* (README) for the problem."""
    err = meta.get("error")
    if err:
        return f"eval_error: {err}"
    cg = meta.get("compile_go_judge")
    if isinstance(cg, dict) and cg.get("status") is not None:
        if cg.get("status") != "Accepted" or not _go_judge_exit_ok(cg.get("exitStatus")):
            st = cg.get("status")
            es = cg.get("exitStatus")
            extra = f" ({cg['error']})" if cg.get("error") else ""
            return f"compile: {st} (exitStatus={es}){extra}"
    fn = meta.get("first_non_accepted")
    if isinstance(fn, dict) and fn.get("status") is not None:
        return f"run: {fn.get('status')} (exitStatus={fn.get('exitStatus')}, test_index={fn.get('test_index')})"
    fw = meta.get("first_wrong_answer")
    if isinstance(fw, dict):
        return f"wrong_answer (test_index={fw.get('test_index')})"
    if not meta.get("num_tests"):
        return "no_tests"
    return "Accepted"


def _empty_go_judge_run_summary(num_tests: int) -> dict[str, Any]:
    return {
        "num_tests": num_tests,
        "run_status_histogram": {},
        "first_non_accepted": None,
        "first_wrong_answer": None,
    }


def _grade_stdio_tests_with_copy_in(
    client: GoJudgeClient,
    tests: list[StdioTestCase],
    time_limit_s: float,
    mem_bytes: int,
    copy_in: dict,
    run_args: list[str],
) -> tuple[list[bool], dict[str, Any]]:
    """Run the same cached binary (or static copyIn) against every test case."""
    out: list[bool] = []
    hist: Counter[str] = Counter()
    first_non_accepted: dict[str, Any] | None = None
    first_wrong_answer: dict[str, Any] | None = None
    for i, tc in enumerate(tests):
        stdout, r0 = _run_one_go_judge_case(client, tc.input, time_limit_s, mem_bytes, copy_in, run_args)
        st = r0.get("status")
        hist[str(st) if st is not None else "null"] += 1
        if not _go_judge_run_ok(r0):
            out.append(False)
            if first_non_accepted is None:
                files = r0.get("files") or {}
                first_non_accepted = {
                    "test_index": i,
                    **_go_judge_result_meta(r0),
                    "stderr_preview": _truncatefn((files.get("stderr") or ""), 400),
                }
            continue
        ok, _ = compare_stdio_to_expected(stdout, tc.output)
        out.append(ok)
        if not ok and first_wrong_answer is None:
            first_wrong_answer = {"test_index": i, **_go_judge_result_meta(r0)}
    summary = {
        "run_status_histogram": dict(hist),
        "first_non_accepted": first_non_accepted,
        "first_wrong_answer": first_wrong_answer,
    }
    return out, summary


def _compile_with_copy_out_cached(
    client: GoJudgeClient,
    source_key: str,
    code: str,
    compile_args: list[str],
    artifact_key: str,
    compile_timeout_s: float,
) -> tuple[str | None, str, dict[str, Any]]:
    ns = seconds_to_go_judge_duration_ns(compile_timeout_s)
    payload = {
        "cmd": [
            {
                "args": compile_args,
                "env": _GO_JUDGE_ENV,
                "files": [
                    {"content": ""},
                    {"name": "stdout", "max": client.compile_stdout_max},
                    {"name": "stderr", "max": client.compile_stderr_max},
                ],
                "cpuLimit": ns,
                "clockLimit": ns,
                "memoryLimit": client.compile_memory_bytes,
                "procLimit": client.proc_limit,
                "copyIn": {source_key: {"content": code}},
                "copyOut": ["stdout", "stderr"],
                "copyOutCached": [artifact_key],
            }
        ]
    }
    err_meta = {"status": None, "exitStatus": None, "error": ""}
    try:
        results = client.run_cmd(payload)
    except Exception as e:
        err_meta["error"] = str(e)
        return None, str(e), err_meta
    if not results:
        err_meta["error"] = "empty go-judge response"
        return None, "empty go-judge response", err_meta
    r0 = results[0]
    result_meta = _go_judge_result_meta(r0)
    stderr = ((r0.get("files") or {}).get("stderr") or "").strip()
    if r0.get("status") != "Accepted" or not _go_judge_exit_ok(r0.get("exitStatus")):
        return None, stderr or (r0.get("error") or r0.get("status") or "compile failed"), result_meta
    fid = (r0.get("fileIds") or {}).get(artifact_key)
    if not fid:
        fail_meta = {**result_meta, "error": result_meta.get("error") or "missing artifact fileId after compile"}
        return None, stderr or "missing artifact fileId after compile", fail_meta
    return fid, stderr, result_meta


def _compile_cpp(
    client: GoJudgeClient, source: str, compile_timeout_s: float
) -> tuple[str | None, str, dict[str, Any]]:
    return _compile_with_copy_out_cached(
        client,
        "solution.cpp",
        source,
        [client.gxx_path, "solution.cpp", "-o", "solution"],
        "solution",
        compile_timeout_s,
    )


def _run_stdio_custom_profile(
    client: GoJudgeClient,
    code: str,
    profile: dict[str, Any],
    tests: list[StdioTestCase],
    time_limit_s: float,
    mem_bytes: int,
    compile_timeout_s: float,
) -> list[bool]:
    """User-defined compile+run (see module docstring for ``language_profiles`` schema)."""
    required = ("source_key", "compile_args", "artifact_key", "run_args")
    for k in required:
        if k not in profile:
            raise ValueError(f'language_profiles entry must include "{k}": {required}')
    source_key = str(profile["source_key"])
    compile_args = list(profile["compile_args"])
    artifact_key = str(profile["artifact_key"])
    run_args = list(profile["run_args"])
    binary_id, cerr, compile_meta = _compile_with_copy_out_cached(
        client, source_key, code, compile_args, artifact_key, compile_timeout_s
    )
    n = len(tests)
    if binary_id is None:
        _log_first_cpp_compile_failure(cerr or "(no stderr)", client=client)
        return [False] * n, {
            **_empty_go_judge_run_summary(n),
            "compile_go_judge": compile_meta,
            "compile_stderr_preview": _truncatefn(cerr, 800),
        }
    copy_key = str(profile.get("run_copy_in_key", artifact_key))
    try:
        per_test, run_summary = _grade_stdio_tests_with_copy_in(
            client,
            tests,
            time_limit_s,
            mem_bytes,
            {copy_key: {"fileId": binary_id}},
            run_args,
        )
    finally:
        client.delete_file(binary_id)
    return per_test, {
        "num_tests": n,
        "compile_go_judge": compile_meta,
        **run_summary,
    }


def run_stdio_tests_with_go_judge(
    client: GoJudgeClient,
    code: str,
    language: str,
    tests: list[StdioTestCase],
    time_limit_s: float,
    memory_limit_mb: int,
    compile_timeout_s: float,
    language_profiles: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[bool], dict[str, Any]]:
    """Run every zip test case; return pass/fail per case and go-judge metadata."""
    n = len(tests)
    if not tests:
        return [], {**_empty_go_judge_run_summary(0), "compile_go_judge": None}
    lang = _normalize_language(language)
    mem_bytes = min(client.run_memory_bytes, max(16 * 1024**2, int(memory_limit_mb) * 1024 * 1024))
    profiles = language_profiles or {}
    if lang in profiles:
        return _run_stdio_custom_profile(
            client, code, profiles[lang], tests, time_limit_s, mem_bytes, compile_timeout_s
        )

    if lang == "cpp":
        binary_id, cerr, compile_meta = _compile_cpp(client, code, compile_timeout_s)
        if binary_id is None:
            _log_first_cpp_compile_failure(cerr or "(no stderr)", client=client)
            return [False] * n, {
                **_empty_go_judge_run_summary(n),
                "compile_go_judge": compile_meta,
                "compile_stderr_preview": _truncatefn(cerr, 800),
            }
        try:
            per_test, run_summary = _grade_stdio_tests_with_copy_in(
                client,
                tests,
                time_limit_s,
                mem_bytes,
                {"solution": {"fileId": binary_id}},
                ["solution"],
            )
        finally:
            client.delete_file(binary_id)
        return per_test, {
            "num_tests": n,
            "compile_go_judge": compile_meta,
            **run_summary,
        }

    if lang == "python":
        out: list[bool] = []
        hist: Counter[str] = Counter()
        first_non_accepted: dict[str, Any] | None = None
        first_wrong_answer: dict[str, Any] | None = None
        for i, tc in enumerate(tests):
            stdout, r0 = _run_one_go_judge_case(
                client,
                tc.input,
                time_limit_s,
                mem_bytes,
                {"solution.py": {"content": code}},
                [client.python_path, "solution.py"],
            )
            st = r0.get("status")
            hist[str(st) if st is not None else "null"] += 1
            if not _go_judge_run_ok(r0):
                out.append(False)
                if first_non_accepted is None:
                    files = r0.get("files") or {}
                    first_non_accepted = {
                        "test_index": i,
                        **_go_judge_result_meta(r0),
                        "stderr_preview": _truncatefn((files.get("stderr") or ""), 400),
                    }
                continue
            ok, _ = compare_stdio_to_expected(stdout, tc.output)
            out.append(ok)
            if not ok and first_wrong_answer is None:
                first_wrong_answer = {"test_index": i, **_go_judge_result_meta(r0)}
        return out, {
            "num_tests": n,
            "compile_go_judge": None,
            "run_status_histogram": dict(hist),
            "first_non_accepted": first_non_accepted,
            "first_wrong_answer": first_wrong_answer,
        }

    raise ValueError(
        f"Unsupported LiveCodeBench-Pro language {language!r}. "
        f"Built-ins: cpp, python (aliases: py, python3). "
        f"Add eval_config.go_judge.language_profiles.{lang} with keys "
        f"source_key, compile_args, artifact_key, run_args (optional run_copy_in_key)."
    )


def _estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    def estimator(n: int, c: int, kk: int) -> float:
        if n - c < kk:
            return 1.0
        return 1.0 - float(np.prod(1.0 - kk / np.arange(n - c + 1, n + 1)))

    return float(estimator(int(num_samples), int(num_correct), int(k)))


def compute_pass_at_k_metrics(results: dict[int, list[list[Any]]], k_list: list[int]) -> dict[str, Any]:
    """``results[i]`` = list of generations; each generation = list of per-test grades (bool or number)."""
    if not results:
        return {}
    total: list[int] = []
    correct: list[int] = []
    task_ids: list[int] = []
    for task_id in sorted(results.keys()):
        res = results[task_id]
        task_ids.append(task_id)
        total.append(len(res))
        correct.append(sum(bool(np.all(np.asarray(generation) > 0)) for generation in res))
    total_a = np.array(total)
    pass_at_k: dict[str, Any] = {}
    detail_pass_at_k: dict[str, dict[int, float]] = {}
    for k in k_list:
        if not (total_a >= k).all():
            continue
        est = [_estimate_pass_at_k(int(n), int(c), k) for n, c in zip(total, correct)]
        pass_at_k[f"pass@{k}"] = float(np.mean(est))
        detail_pass_at_k[f"pass@{k}"] = dict(zip(task_ids, est))
    pass_at_k["detail"] = detail_pass_at_k
    return pass_at_k


def extract_instance_pass_list(results: dict[int, list[list[Any]]]) -> list[list[bool]]:
    """One list per problem (benchmark order by sorted task id): bool per generation."""
    graded: list[list[bool]] = []
    for task_id in sorted(results.keys()):
        res = results[task_id]
        row = []
        for generation in res:
            row.append(all(g > 0 for g in generation))
        graded.append(row)
    return graded


def _build_eval_row(
    row: dict,
    code_list: list[str],
    graded_per_generation: list[bool],
    return_status: str,
    num_tests: int,
) -> dict[str, Any]:
    difficulty = row.get("difficulty", "medium")
    if hasattr(difficulty, "value"):
        difficulty = difficulty.value
    platform = row.get("platform", "codeforces")
    if hasattr(platform, "value"):
        platform = platform.value
    return {
        "platform": platform,
        "problem_id": row["problem_id"],
        "problem_title": row.get("problem_title", ""),
        "problem_statement": row.get("problem_statement", ""),
        "link": row.get("link", ""),
        "time_limit": row.get("time_limit", 1),
        "memory_limit": row.get("memory_limit", 256),
        "difficulty": difficulty,
        "output_list": code_list,
        "code_list": code_list,
        "language": row.get("language", "cpp"),
        "graded_list": graded_per_generation,
        "pass@1": graded_per_generation.count(True) / len(graded_per_generation) if graded_per_generation else 0.0,
        "return_status": return_status,
        "metadata": [json.dumps({"num_tests": num_tests})],
    }


_tls = threading.local()
_compile_fail_lock = threading.Lock()
_compile_fail_logged = False


def _reset_lcb_pro_compile_fail_log() -> None:
    global _compile_fail_logged
    with _compile_fail_lock:
        _compile_fail_logged = False


def _log_first_cpp_compile_failure(msg: str, *, client: GoJudgeClient | None = None) -> None:
    global _compile_fail_logged
    with _compile_fail_lock:
        if _compile_fail_logged:
            return
        _compile_fail_logged = True
    truncated = _truncatefn(msg, 1500)
    conn_hint = ""
    if "Connection refused" in msg or "Errno 111" in msg:
        ep = client.base_url if client is not None else "http://127.0.0.1:5050"
        conn_hint = (
            f" Connection refused to {ep}: go-judge is not listening. "
            "For auto-start: use a nemo-skills image built with go-judge (Dockerfile.nemo-skills) and packaged code "
            "that defines livecodebench-pro INSTALLATION_COMMAND; check Slurm logs for the install step and /tmp/go-judge.log. "
            "If the binary is missing, the job should fail at install—otherwise go-judge may have exited (e.g. cgroup read-only in Pyxis). "
            "Otherwise run go-judge externally and set NEMO_SKILLS_GO_JUDGE_HOST / _PORT (and NEMO_SKILLS_SKIP_AUTO_GO_JUDGE=1)."
        )
    LOG.warning(
        "LiveCodeBench-Pro: first C++ compile failure via go-judge "
        "(check g++ in judge env, empty code, wrong language, or judge unreachable). stderr: %s%s",
        truncated,
        conn_hint,
    )


def _thread_local_client(cfg: dict) -> GoJudgeClient:
    c = getattr(_tls, "gj", None)
    if c is None:
        c = GoJudgeClient.from_config(cfg)
        _tls.gj = c
    return c


def run_livecodebench_pro_evaluation(
    *,
    jsonl_path: str,
    test_dir: str,
    language: str,
    go_judge: dict,
    num_workers: int = 12,
    timeout: int = 6,
    k_list: list[int] | None = None,
    compile_timeout_s: float = 60.0,
) -> None:
    """Load zips + generations JSONL, grade via go-judge, write ``<stem>_eval_results.json``."""
    k_list = k_list or [1]
    gj_cfg = dict(go_judge or {})
    _reset_lcb_pro_compile_fail_log()
    zip_index = index_zip_files(Path(test_dir))

    benchmark_rows: list[dict] = []
    skipped_zip = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("problem_id", ""))
            if not zip_index.get(pid):
                skipped_zip += 1
                continue
            benchmark_rows.append(row)

    if skipped_zip:
        LOG.warning("Skipped %d JSONL rows with no matching {problem_id}.zip under %s", skipped_zip, test_dir)

    out_path = _eval_results_json_path(jsonl_path)
    if not benchmark_rows:
        LOG.warning("No LiveCodeBench-Pro problems to evaluate (check JSONL and zip directory)")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "eval": {},
                    "detail_pass@1": {},
                },
                f,
                indent=4,
            )
        return

    LOG.info("Evaluating %d LiveCodeBench-Pro problems (go-judge)", len(benchmark_rows))

    results_linear: dict[int, list[list[Any]]] = {}
    return_status_linear: dict[int, str] = {}
    num_tests_linear: dict[int, int] = {}
    language_profiles = gj_cfg.get("language_profiles") or {}

    def work(item: tuple[int, dict]) -> None:
        idx, row = item
        pid = str(row["problem_id"])
        tests: list[StdioTestCase] = []
        try:
            code_list = row.get("code_list") or [row.get("completion", "")]
            code = code_list[0] if code_list else ""
            zp = zip_index[pid]
            tests = read_stdio_tests_from_zip(zp)
            tl = min(int(row.get("time_limit", timeout)), int(timeout))
            mem_mb = int(row.get("memory_limit", 256))
            client = _thread_local_client(gj_cfg)
            row_lang = _normalize_language(str(row.get("language") or language))
            per_test, judge_meta = run_stdio_tests_with_go_judge(
                client,
                code,
                row_lang,
                tests,
                float(tl),
                mem_mb,
                compile_timeout_s=float(max(compile_timeout_s, float(tl))),
                language_profiles=language_profiles,
            )
            results_linear[idx] = [per_test]
            num_tests_linear[idx] = len(tests)
            return_status_linear[idx] = _summarize_return_status(judge_meta)
        except Exception as e:
            LOG.exception("LCB-Pro go-judge eval failed for problem_id=%s", pid)
            results_linear[idx] = [[False]]
            num_tests_linear[idx] = len(tests)
            return_status_linear[idx] = _summarize_return_status({"error": str(e), "num_tests": len(tests)})

    with tqdm(total=len(benchmark_rows), desc="LCB-Pro (go-judge)") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
            futs = [ex.submit(work, (i, r)) for i, r in enumerate(benchmark_rows)]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
                pbar.update(1)

    n = len(benchmark_rows)
    ordered_results = {i: results_linear.get(i, [[False]]) for i in range(n)}
    metrics = compute_pass_at_k_metrics(ordered_results, k_list)
    graded_rows = extract_instance_pass_list(ordered_results)

    save_eval = []
    for i, row in enumerate(benchmark_rows):
        code_list = row.get("code_list") or [row.get("completion", "")]
        gl = graded_rows[i]
        rs = return_status_linear.get(i, "unknown")
        nt = num_tests_linear.get(i, 0)
        save_eval.append(_build_eval_row(row, code_list, gl, rs, nt))

    output_results: dict[str, Any] = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": {r["problem_id"]: r for r in save_eval},
        "detail_pass@1": {},
    }
    for k, v in metrics.items():
        if k == "detail":
            continue
        if isinstance(v, (int, float)):
            LOG.info("%s: %s", k, v)
            output_results[k] = v

    diff_buckets: dict[str, list[float]] = {}
    for r in save_eval:
        d = r["difficulty"]
        diff_buckets.setdefault(d, []).append(float(r["pass@1"]))
    for tag, vals in diff_buckets.items():
        output_results["detail_pass@1"][tag] = sum(vals) / len(vals) if vals else 0.0
        LOG.info("%s pass@1: %s", tag, output_results["detail_pass@1"][tag])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_results, f, indent=4)


def clear_thread_local_go_judge_client() -> None:
    if hasattr(_tls, "gj"):
        try:
            _tls.gj.close()
        except Exception:
            pass
        delattr(_tls, "gj")


@nested_dataclass(kw_only=True)
class LiveCodeBenchProEvaluatorConfig(BaseEvaluatorConfig):
    go_judge: dict | None = None
    language: str = "cpp"
    test_dir: str = None
    timeout: int = 6
    num_processes: int = 12
    compile_timeout_s: float = 60.0


def _as_plain_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return dict(obj)


def _merged_go_judge_client_config(go_judge: Any, sandbox: Any) -> dict:
    """Merge ``go_judge`` with optional legacy ``sandbox`` block (``sandbox_type: go_judge``)."""

    merged: dict = {}
    merged.update(_as_plain_dict(go_judge))
    sb = _as_plain_dict(sandbox)
    if str(sb.get("sandbox_type", "")).lower() == "go_judge":
        merged.update({k: v for k, v in sb.items() if k != "sandbox_type"})
    if merged.get("enabled") is False:
        raise ValueError("LiveCodeBench-Pro evaluation uses go-judge only. Remove go_judge.enabled=false or omit it.")
    return merged


def eval_livecodebench_pro(cfg):
    raw = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    raw["go_judge"] = _merged_go_judge_client_config(raw.get("go_judge"), raw.pop("sandbox", None))
    cfg = LiveCodeBenchProEvaluatorConfig(**raw)
    jsonl_file = cfg.input_file
    samples = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            # preprocess_code only reads "generation"; backfill from "completion" for this benchmark only.
            gen = sample.get("generation")
            if not isinstance(gen, str) or not gen.strip():
                comp = sample.get("completion")
                if isinstance(comp, str) and comp.strip():
                    sample = dict(sample)
                    sample["generation"] = comp
            fence_lang = _normalize_language(str(sample.get("language") or cfg.language))
            sample = preprocess_code(sample, language=fence_lang, strip_whitespace=True)
            sample["code_list"] = [sample["completion"]]
            samples.append(sample)

    with open(jsonl_file, "wt", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    go_judge_cfg = dict(cfg.go_judge or {})
    results_path = _eval_results_json_path(jsonl_file)
    saved_path = results_path[:-5] + "-saved.json" if results_path.endswith(".json") else results_path + "-saved"

    try:
        run_livecodebench_pro_evaluation(
            jsonl_path=jsonl_file,
            test_dir=cfg.test_dir,
            language=cfg.language,
            go_judge=go_judge_cfg,
            num_workers=cfg.num_processes,
            timeout=cfg.timeout,
            k_list=[1],
            compile_timeout_s=cfg.compile_timeout_s,
        )
    finally:
        clear_thread_local_go_judge_client()

    with open(results_path, encoding="utf-8") as fin:
        eval_grades = json.load(fin)
    with open(jsonl_file, "wt", encoding="utf-8") as f:
        for sample in samples:
            if sample["problem_id"] in eval_grades["eval"]:
                ev = eval_grades["eval"][sample["problem_id"]]
                sample["graded_list"] = ev["graded_list"]
                sample["return_status"] = ev.get("return_status", "")
                sample["metadata"] = ev.get("metadata", [json.dumps({"num_tests": 0})])
                f.write(json.dumps(sample) + "\n")

    shutil.move(results_path, saved_path)
