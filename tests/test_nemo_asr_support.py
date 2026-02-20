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

import asyncio
import io
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

from nemo_skills.inference.model.nemo_asr import NemoASRModel
from nemo_skills.inference.server.serve_nemo_asr import NemoASRServer
from nemo_skills.pipeline.utils.server import SupportedServers


def _close_model(model: NemoASRModel):
    asyncio.run(model._client.aclose())


def test_nemo_asr_resolve_audio_path_from_tarred_member(tmp_path):
    member_name = "sample.wav"
    member_bytes = b"fake wav content"
    tar_path = tmp_path / "shard_0.tar"

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(member_bytes)
        tar.addfile(info, fileobj=io.BytesIO(member_bytes))

    model = NemoASRModel(data_dir=str(tmp_path), tarred_audio_filepaths="shard_0.tar")
    extracted_path, cleanup_path = model._resolve_audio_path(member_name)
    try:
        assert extracted_path.is_file()
        assert cleanup_path == extracted_path
        assert extracted_path.read_bytes() == member_bytes
    finally:
        extracted_path.unlink(missing_ok=True)
        _close_model(model)


def test_nemo_asr_resolve_audio_path_from_explicit_tar_prompt(tmp_path):
    member_name = "nested/sample.wav"
    member_bytes = b"fake wav content 2"
    tar_path = tmp_path / "shard_1.tar"

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(member_bytes)
        tar.addfile(info, fileobj=io.BytesIO(member_bytes))

    model = NemoASRModel(data_dir=str(tmp_path))
    extracted_path, cleanup_path = model._resolve_audio_path(f"shard_1.tar::{member_name}")
    try:
        assert extracted_path.is_file()
        assert cleanup_path == extracted_path
        assert extracted_path.read_bytes() == member_bytes
    finally:
        extracted_path.unlink(missing_ok=True)
        _close_model(model)


def test_nemo_asr_server_extract_first_hypothesis_validation():
    hypothesis = SimpleNamespace(text="hello")
    assert NemoASRServer._extract_first_hypothesis([hypothesis]) is hypothesis
    assert NemoASRServer._extract_first_hypothesis([[hypothesis]]) is hypothesis

    try:
        NemoASRServer._extract_first_hypothesis([])
        raise AssertionError("Expected RuntimeError for empty hypotheses")
    except RuntimeError as e:
        assert "No transcription returned" in str(e)

    try:
        NemoASRServer._extract_first_hypothesis([[]])
        raise AssertionError("Expected RuntimeError for empty inner hypotheses")
    except RuntimeError as e:
        assert "empty transcription hypotheses" in str(e)


def test_nemo_asr_server_transcribe_single_passes_language_id_when_supported():
    class _FakeModel:
        def transcribe(self, paths, batch_size, return_hypotheses, timestamps, language_id=None):
            assert paths == ["a.wav"]
            assert batch_size == 1
            assert return_hypotheses is True
            assert timestamps is False
            assert language_id == "el"
            return [[SimpleNamespace(text="ok")]]

    server = NemoASRServer.__new__(NemoASRServer)
    server.model = _FakeModel()
    output = server._transcribe_single(["a.wav"], enable_timestamps=False, language="el")
    assert output[0][0].text == "ok"


def test_nemo_asr_server_transcribe_single_without_language_id_param():
    class _FakeModel:
        def transcribe(self, paths, batch_size, return_hypotheses, timestamps):
            assert paths == ["a.wav"]
            return [SimpleNamespace(text="ok")]

    server = NemoASRServer.__new__(NemoASRServer)
    server.model = _FakeModel()
    output = server._transcribe_single(["a.wav"], enable_timestamps=True, language="el")
    assert output[0].text == "ok"


def test_nemo_asr_resolve_audio_path_from_s3_tar_prompt(tmp_path):
    member_name = "sample.wav"
    member_bytes = b"fake wav content s3"
    local_tar = tmp_path / "cached_s3_shard.tar"
    s3_tar_uri = "s3://my-bucket/audio_0.tar"

    with tarfile.open(local_tar, "w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(member_bytes)
        tar.addfile(info, fileobj=io.BytesIO(member_bytes))

    model = NemoASRModel(data_dir=str(tmp_path))
    try:
        with (
            patch.object(NemoASRModel, "_is_datastore_path", side_effect=lambda p: p.startswith("s3://")),
            patch.object(NemoASRModel, "_download_datastore_object", return_value=local_tar),
        ):
            extracted_path, cleanup_path = model._resolve_audio_path(f"{s3_tar_uri}:{member_name}")
            assert extracted_path.is_file()
            assert cleanup_path == extracted_path
            assert extracted_path.read_bytes() == member_bytes
            extracted_path.unlink(missing_ok=True)
    finally:
        _close_model(model)


def test_nemo_asr_extract_audio_reference_from_openai_messages(tmp_path):
    model = NemoASRModel(data_dir=str(tmp_path))
    try:
        prompt = [
            {"role": "system", "content": "Answer the questions."},
            {
                "role": "user",
                "content": "Transcribe",
                "audio": {"path": "/data/example.wav", "duration": 1.0},
                "audios": [{"path": "/data/example.wav", "duration": 1.0}],
            },
        ]
        assert model._extract_audio_reference(prompt) == "/data/example.wav"
    finally:
        _close_model(model)


def test_nemo_asr_extract_audio_reference_from_native_manifest_dict(tmp_path):
    model = NemoASRModel(data_dir=str(tmp_path))
    try:
        prompt = {
            "audio_filepath": "/data/native.wav",
            "duration": 1.23,
            "text": "hello",
        }
        assert model._extract_audio_reference(prompt) == "/data/native.wav"

        prompt_audio_path = {
            "audio_path": ["/data/list_based.wav"],
            "text": "hello",
        }
        assert model._extract_audio_reference(prompt_audio_path) == "/data/list_based.wav"

        prompt_audio_filename = {
            "audio_filename": "relative_name.wav",
            "duration": 1.0,
        }
        assert model._extract_audio_reference(prompt_audio_filename) == "relative_name.wav"

        prompt_context = {
            "context": "ais://bucket/path/audio.wav",
            "duration": 1.0,
        }
        assert model._extract_audio_reference(prompt_context) == "ais://bucket/path/audio.wav"
    finally:
        _close_model(model)


def test_nemo_asr_resolve_audio_path_from_shard_id_optimized(tmp_path):
    member_name = "manifest_member.flac"
    member_bytes = b"fake wav content shard"
    tar_path = tmp_path / "audio_9.tar"

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(member_bytes)
        tar.addfile(info, fileobj=io.BytesIO(member_bytes))

    model = NemoASRModel(data_dir=str(tmp_path), tarred_audio_filepaths="audio_*.tar")
    try:
        prompt = {"audio_filepath": member_name, "shard_id": 9}
        audio_reference = model._extract_audio_reference(prompt)
        shard_tar_path = model._resolve_tar_path_from_shard_id(model._extract_shard_id(prompt))
        extracted_path = model._extract_audio_member(shard_tar_path, audio_reference)
        assert extracted_path.is_file()
        assert extracted_path.read_bytes() == member_bytes
        extracted_path.unlink(missing_ok=True)
    finally:
        _close_model(model)


def test_nemo_asr_shard_id_does_not_override_explicit_tar_member_prompt(tmp_path):
    member_name = "member_in_tar.flac"
    member_bytes = b"member bytes"
    local_tar = tmp_path / "audio_9.tar"
    s3_tar_uri = "s3://my-bucket/audio_9.tar"

    with tarfile.open(local_tar, "w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(member_bytes)
        tar.addfile(info, fileobj=io.BytesIO(member_bytes))

    model = NemoASRModel(data_dir=str(tmp_path), tarred_audio_filepaths=str(local_tar))
    try:
        prompt = {"audio_filepath": f"{s3_tar_uri}:{member_name}", "shard_id": 9}
        with (
            patch.object(NemoASRModel, "_is_datastore_path", side_effect=lambda p: p.startswith("s3://")),
            patch.object(NemoASRModel, "_download_datastore_object", return_value=local_tar),
        ):
            extracted_path, cleanup_path = model._resolve_audio_from_prompt(prompt)
            assert extracted_path.is_file()
            assert cleanup_path == extracted_path
            assert extracted_path.read_bytes() == member_bytes
            extracted_path.unlink(missing_ok=True)
    finally:
        _close_model(model)


def test_supported_servers_includes_nemo_asr():
    assert SupportedServers.nemo_asr.value == "nemo_asr"


def test_nemo_asr_base_url_with_scheme_port_and_path(tmp_path):
    model = NemoASRModel(base_url="https://host.example:5000/v1", data_dir=str(tmp_path))
    try:
        assert model.server_host == "host.example"
        assert model.server_port == "5000"
        assert model.base_url == "https://host.example:5000/v1"
    finally:
        _close_model(model)
