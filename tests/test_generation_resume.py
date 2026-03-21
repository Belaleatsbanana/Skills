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

import json

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig


def test_skip_completed_samples_rerun_soft_failed_errors_preserves_completed_rows(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    async_output_file = tmp_path / "output.jsonl-async"

    input_rows = [{"id": 0}, {"id": 1}, {"id": 2}]
    input_file.write_text("".join(json.dumps(row) + "\n" for row in input_rows), encoding="utf-8")

    existing_rows = [
        {"generation": "first ok", "id": 0},
        {"generation": "", "id": 1, "detailed_error": "RateLimitError: retry exhausted"},
        {"generation": "third ok", "id": 2},
    ]
    output_file.write_text("".join(json.dumps(row) + "\n" for row in existing_rows), encoding="utf-8")

    cfg = GenerationTaskConfig(
        input_file=str(input_file),
        output_file=str(output_file),
        prompt_format="openai",
        server={"server_type": "openai", "model": "dummy"},
        sandbox={},
        skip_filled=True,
        rerun_soft_failed_errors=True,
    )
    task = GenerationTask.__new__(GenerationTask)
    task.cfg = cfg

    remaining = task.skip_completed_samples([row.copy() for row in input_rows])

    assert [row[task.cfg.async_position_key] for row in remaining] == [1]
    assert [row["id"] for row in remaining] == [1]

    preserved_rows = [json.loads(line) for line in async_output_file.read_text(encoding="utf-8").splitlines()]
    assert [row[task.cfg.async_position_key] for row in preserved_rows] == [0, 2]
    assert [row["generation"] for row in preserved_rows] == ["first ok", "third ok"]



def test_skip_completed_samples_without_rerun_soft_failed_errors_skips_finished_output(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    input_rows = [{"id": 0}, {"id": 1}]
    input_file.write_text("".join(json.dumps(row) + "\n" for row in input_rows), encoding="utf-8")
    output_rows = [
        {"generation": "ok", "id": 0},
        {"generation": "", "id": 1, "detailed_error": "RateLimitError: retry exhausted"},
    ]
    output_file.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")

    cfg = GenerationTaskConfig(
        input_file=str(input_file),
        output_file=str(output_file),
        prompt_format="openai",
        server={"server_type": "openai", "model": "dummy"},
        sandbox={},
        skip_filled=True,
        rerun_soft_failed_errors=False,
    )
    task = GenerationTask.__new__(GenerationTask)
    task.cfg = cfg

    remaining = task.skip_completed_samples([row.copy() for row in input_rows])

    assert remaining == []



def test_skip_completed_samples_rerun_soft_failed_errors_uses_async_output(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    async_output_file = tmp_path / "output.jsonl-async"

    input_rows = [{"id": 0}, {"id": 1}, {"id": 2}]
    input_file.write_text("".join(json.dumps(row) + "\n" for row in input_rows), encoding="utf-8")

    async_rows = [
        {"generation": "first ok", "id": 0, "_async_position": 0},
        {"generation": "", "id": 1, "_async_position": 1, "detailed_error": "RateLimitError: retry exhausted"},
        {"generation": "third ok", "id": 2, "_async_position": 2},
    ]
    async_output_file.write_text("".join(json.dumps(row) + "\n" for row in async_rows), encoding="utf-8")

    cfg = GenerationTaskConfig(
        input_file=str(input_file),
        output_file=str(output_file),
        prompt_format="openai",
        server={"server_type": "openai", "model": "dummy"},
        sandbox={},
        skip_filled=True,
        rerun_soft_failed_errors=True,
    )
    task = GenerationTask.__new__(GenerationTask)
    task.cfg = cfg

    remaining = task.skip_completed_samples([row.copy() for row in input_rows])

    assert [row[task.cfg.async_position_key] for row in remaining] == [1]
    assert [row["id"] for row in remaining] == [1]

    preserved_rows = [json.loads(line) for line in async_output_file.read_text(encoding="utf-8").splitlines()]
    assert [row[task.cfg.async_position_key] for row in preserved_rows] == [0, 2]
    assert [row["generation"] for row in preserved_rows] == ["first ok", "third ok"]



def test_skip_completed_samples_rerun_soft_failed_errors_resumes_clean_async_output(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    async_output_file = tmp_path / "output.jsonl-async"

    input_rows = [{"id": 0}, {"id": 1}, {"id": 2}]
    input_file.write_text("".join(json.dumps(row) + "\n" for row in input_rows), encoding="utf-8")

    async_rows = [
        {"generation": "first ok", "id": 0, "_async_position": 0},
        {"generation": "second ok", "id": 1, "_async_position": 1},
    ]
    async_output_file.write_text("".join(json.dumps(row) + "\n" for row in async_rows), encoding="utf-8")

    cfg = GenerationTaskConfig(
        input_file=str(input_file),
        output_file=str(output_file),
        prompt_format="openai",
        server={"server_type": "openai", "model": "dummy"},
        sandbox={},
        skip_filled=True,
        rerun_soft_failed_errors=True,
    )
    task = GenerationTask.__new__(GenerationTask)
    task.cfg = cfg

    remaining = task.skip_completed_samples([row.copy() for row in input_rows])

    assert [row[task.cfg.async_position_key] for row in remaining] == [2]
    assert [row["id"] for row in remaining] == [2]
