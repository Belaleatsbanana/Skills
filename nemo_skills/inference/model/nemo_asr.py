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

"""NeMo ASR model client for connecting to serve_nemo_asr.py server.

This client wraps the NeMo ASR server with the BaseModel API pattern,
allowing it to be used in the nemo-skills inference pipeline.
"""

from __future__ import annotations

import glob
import logging
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import aiofiles
import httpx

from nemo_skills.utils import get_logger_name

from .nim_utils import setup_ssh_tunnel, validate_unsupported_params

LOG = logging.getLogger(get_logger_name(__file__))


class NemoASRModel:
    """Client wrapper for NeMo ASR server.

    This model follows the BaseModel pattern and connects to a NeMo ASR server
    running serve_nemo_asr.py via HTTP.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Hostname or IP of the NeMo ASR server.
    port : str, default "5000"
        HTTP port of the server.
    model : str, default "nemo-asr"
        Model identifier (mostly for compatibility, server uses its loaded model).
    base_url : str | None
        Full base URL (overrides host:port if provided).
    language_code : str, default "en"
        Language code for recognition.
    response_format : str, default "verbose_json"
        Response format: "json" or "verbose_json".
    enable_timestamps : bool, default True
        Whether to request word-level timestamps.
    enable_audio_chunking : bool, default True
        Whether to automatically chunk long audio files.
    chunk_audio_threshold_sec : int, default 30
        Audio duration threshold (in seconds) for automatic chunking.
    ssh_server : str | None
        SSH server for tunneling (format: [user@]host).
    ssh_key_path : str | None
        Path to SSH key for tunneling.
    tokenizer : str | None
        Accepted for API compatibility; ignored by NemoASRModel.
    data_dir : str, default ""
        Base directory for resolving relative audio paths.
    output_dir : str | None
        Accepted for API compatibility; ignored by NemoASRModel.
    tarred_audio_filepaths : str | list[str] | None
        Optional path(s) to tar shards for NeMo tarred ASR datasets.
        If configured, `prompt` can be a member filename from tarred manifest.
    max_workers : int, default 64
        Maximum concurrent requests.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "5000",
        model: str = "nemo-asr",
        *,
        base_url: str | None = None,
        language_code: str = "en",
        response_format: str = "verbose_json",
        enable_timestamps: bool = True,
        enable_audio_chunking: bool = True,
        chunk_audio_threshold_sec: int = 30,
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
        tokenizer: str | None = None,
        data_dir: str = "",
        output_dir: str | None = None,
        tarred_audio_filepaths: str | list[str] | None = None,
        max_workers: int = 64,
    ) -> None:
        """Initialize NemoASRModel client."""
        if tokenizer is not None:
            LOG.warning("NemoASRModel does not use tokenizer. Ignoring tokenizer argument.")
        if output_dir is not None:
            LOG.warning("NemoASRModel does not use output_dir. Ignoring output_dir argument.")

        base_path = ""
        base_scheme = "http"
        # Handle base_url compatibility
        if base_url:
            parsed = urlparse(base_url)
            if parsed.scheme:
                if parsed.hostname:
                    host = parsed.hostname
                if parsed.port is not None:
                    port = str(parsed.port)
                base_scheme = parsed.scheme
                base_path = parsed.path.rstrip("/")
            else:
                _url = base_url.replace("http://", "").replace("https://", "")
                if ":" in _url:
                    host, port = _url.split(":", 1)
                else:
                    host = _url

        # Setup SSH tunnel if needed
        host, port, self._tunnel = setup_ssh_tunnel(host, port, ssh_server, ssh_key_path)

        # Store attributes expected by inference code
        self.model_name_or_path = model
        self.server_host = host
        self.server_port = port
        self.data_dir = data_dir
        self.output_dir = output_dir

        # ASR-specific config
        self.language_code = language_code
        self.response_format = response_format
        self.enable_timestamps = enable_timestamps
        self.enable_audio_chunking = enable_audio_chunking
        self.chunk_audio_threshold_sec = chunk_audio_threshold_sec
        self.tarred_audio_files = self._resolve_tarred_audio_files(tarred_audio_filepaths)
        self._tar_member_index: Dict[str, Path] = {}
        self._tar_local_cache: Dict[str, Path] = {}

        # Build base URL
        if base_path:
            self.base_url = urlunparse((base_scheme, f"{host}:{port}", base_path, "", "", ""))
        else:
            self.base_url = f"{base_scheme}://{host}:{port}"

        # Create HTTP client with connection pooling
        limits = httpx.Limits(max_keepalive_connections=max_workers, max_connections=max_workers)
        self._client = httpx.AsyncClient(limits=limits, timeout=300.0)  # 5 min timeout

        LOG.info(f"Initialized NemoASRModel connecting to {self.base_url}")

    @staticmethod
    def _is_datastore_path(path: str) -> bool:
        """Check if path is a datastore URI (e.g. ais://bucket/object)."""
        try:
            from nemo.utils.data_utils import is_datastore_path

            return is_datastore_path(path)
        except ImportError:
            parsed = urlparse(path)
            return bool(parsed.scheme) and bool(parsed.netloc)

    @staticmethod
    def _download_datastore_object(store_path: str) -> Path:
        """Download datastore object to local cache and return local path."""
        scheme = urlparse(store_path).scheme
        try:
            from nemo.utils.data_utils import DataStoreObject, open_best, resolve_cache_dir
        except ImportError as e:
            raise RuntimeError(
                "Datastore path support requires nemo_toolkit installation. "
                "Install nemo_toolkit to use ais:// or other datastore URIs."
            ) from e

        # Keep AIStore behavior aligned with NeMo's native DataStoreObject path.
        if scheme == "ais":
            local_path = DataStoreObject(store_path).get()
            if local_path is None:
                raise RuntimeError(f"Failed to download datastore object: {store_path}")
            return Path(local_path)

        # For non-AIS datastore URIs (e.g., s3://), use NeMo open_best
        # (typically backed by Lhotse) and cache file locally.
        parsed = urlparse(store_path)
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"Invalid datastore path format: {store_path}")
        rel_path = Path(parsed.netloc + parsed.path)
        local_path = resolve_cache_dir() / "datastore" / scheme / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open_best(store_path, mode="rb") as stream:
            data = stream.read()
        with open(local_path, "wb") as fout:
            fout.write(data)
        return local_path

    def _resolve_tarred_audio_files(self, tarred_audio_filepaths: str | list[str] | None) -> list[str]:
        """Resolve configured tar paths into an explicit list of tar file references."""
        if tarred_audio_filepaths is None:
            return []

        if isinstance(tarred_audio_filepaths, str):
            candidates = [x.strip() for x in tarred_audio_filepaths.split(",") if x.strip()]
        else:
            candidates = [str(x).strip() for x in tarred_audio_filepaths if str(x).strip()]

        tar_files: list[str] = []
        for candidate in candidates:
            if self._is_datastore_path(candidate):
                if not candidate.endswith(".tar"):
                    raise ValueError(f"Datastore tar path must end with .tar: {candidate}")
                tar_files.append(candidate)
                continue

            candidate_path = Path(candidate).expanduser()
            if not candidate_path.is_absolute():
                candidate_path = (Path(self.data_dir) / candidate_path).expanduser()
            matches = glob.glob(str(candidate_path))
            if len(matches) == 0:
                if candidate_path.is_file():
                    matches = [str(candidate_path)]
                else:
                    raise FileNotFoundError(f"Tarred audio path not found: {candidate}")
            for path_str in sorted(matches):
                path = Path(path_str).expanduser()
                if path.suffix == ".tar":
                    tar_files.append(str(path.absolute()))

        tar_files = list(dict.fromkeys(tar_files))
        if tar_files:
            LOG.info(f"Configured {len(tar_files)} tarred audio shard(s) for NemoASRModel")
        return tar_files

    def _materialize_tar_path(self, tar_ref: str) -> Path:
        """Resolve tar reference (local path or datastore URI) to a local filesystem path."""
        if tar_ref in self._tar_local_cache:
            return self._tar_local_cache[tar_ref]

        if self._is_datastore_path(tar_ref):
            local_tar_path = self._download_datastore_object(tar_ref)
        else:
            local_tar_path = Path(tar_ref).expanduser()
            if local_tar_path.is_file():
                pass
            elif not local_tar_path.is_absolute():
                local_tar_path = (Path(self.data_dir) / local_tar_path).expanduser()
            if not local_tar_path.is_file():
                raise FileNotFoundError(f"Tar file not found: {local_tar_path}")

        self._tar_local_cache[tar_ref] = local_tar_path
        return local_tar_path

    def _resolve_tar_member(self, member_name: str) -> Path:
        """Find tar file containing member by scanning configured tar shards."""
        if member_name in self._tar_member_index:
            return self._tar_member_index[member_name]

        for tar_ref in self.tarred_audio_files:
            tar_path = self._materialize_tar_path(tar_ref)
            with tarfile.open(tar_path, "r") as tar:
                try:
                    member = tar.getmember(member_name)
                except KeyError:
                    continue
                if member.isfile():
                    self._tar_member_index[member_name] = tar_path
                    return tar_path

        raise FileNotFoundError(
            f"Audio member '{member_name}' was not found in configured tarred_audio_filepaths: "
            f"{[str(p) for p in self.tarred_audio_files]}"
        )

    @staticmethod
    def _extract_audio_member(tar_path: Path, member_name: str) -> Path:
        """Extract a single audio member from tar to a temporary file."""
        suffix = Path(member_name).suffix or ".wav"
        with tarfile.open(tar_path, "r") as tar:
            file_obj = tar.extractfile(member_name)
            if file_obj is None:
                raise FileNotFoundError(f"Audio member '{member_name}' not found in tar file '{tar_path}'")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_obj.read())
                return Path(tmp_file.name)

    def _resolve_audio_path(self, prompt: str) -> tuple[Path, Path | None]:
        """Resolve prompt to a local audio file, extracting tar member when needed."""
        if (
            self._is_datastore_path(prompt)
            and "::" not in prompt
            and ".tar:" not in prompt
            and not prompt.endswith(".tar")
        ):
            # Non-tar datastore object (e.g., ais://bucket/audio.wav or s3://bucket/audio.wav)
            return self._download_datastore_object(prompt), None

        audio_path = Path(prompt).expanduser()
        if not audio_path.is_absolute():
            audio_path = (Path(self.data_dir) / audio_path).expanduser()
        if audio_path.is_file():
            return audio_path, None

        tar_part, member_name = None, None
        if "::" in prompt:
            tar_part, member_name = prompt.split("::", 1)
        elif ".tar:" in prompt:
            tar_part, member_name = prompt.rsplit(":", 1)
        if tar_part is not None and member_name is not None:
            tar_path = self._materialize_tar_path(tar_part)
            extracted_path = self._extract_audio_member(tar_path, member_name)
            return extracted_path, extracted_path

        if self.tarred_audio_files:
            tar_path = self._resolve_tar_member(prompt)
            extracted_path = self._extract_audio_member(tar_path, prompt)
            return extracted_path, extracted_path

        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    @staticmethod
    def _extract_shard_id(prompt) -> int | None:
        """Extract shard_id from prompt dict if present."""
        if isinstance(prompt, dict) and "shard_id" in prompt:
            shard_id = prompt["shard_id"]
            if isinstance(shard_id, int):
                return shard_id
            if isinstance(shard_id, str) and shard_id.isdigit():
                return int(shard_id)
        return None

    def _resolve_tar_path_from_shard_id(self, shard_id: int) -> Path | None:
        """Resolve shard_id to matching audio_{shard_id}.tar from configured tar shards."""
        expected_name = f"audio_{shard_id}.tar"
        for tar_ref in self.tarred_audio_files:
            if tar_ref.endswith(expected_name):
                return self._materialize_tar_path(tar_ref)
        return None

    def _resolve_audio_from_prompt(self, prompt) -> tuple[Path, Path | None]:
        """Resolve any supported prompt structure to local audio path."""
        audio_reference = self._extract_audio_reference(prompt)
        shard_id = self._extract_shard_id(prompt)

        # Explicit tar-member refs already encode the shard; use generic parsing path.
        is_explicit_tar_member_ref = ("::" in audio_reference) or (".tar:" in audio_reference)
        if shard_id is not None and self.tarred_audio_files and not is_explicit_tar_member_ref:
            shard_tar_path = self._resolve_tar_path_from_shard_id(shard_id)
            if shard_tar_path is not None:
                extracted_path = self._extract_audio_member(shard_tar_path, audio_reference)
                return extracted_path, extracted_path

        return self._resolve_audio_path(audio_reference)

    @staticmethod
    def _extract_audio_path_from_message(message: dict) -> str | None:
        """Extract first audio path from a single OpenAI-style message dict."""
        if "audios" in message and isinstance(message["audios"], list) and len(message["audios"]) > 0:
            first_audio = message["audios"][0]
            if isinstance(first_audio, dict) and "path" in first_audio:
                return first_audio["path"]
        if "audio" in message and isinstance(message["audio"], dict) and "path" in message["audio"]:
            return message["audio"]["path"]
        return None

    def _extract_audio_reference(self, prompt) -> str:
        """Extract audio reference string from supported prompt structures."""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str):
            return prompt[0]

        # OpenAI-style prompt as list of messages
        if isinstance(prompt, list):
            for message in prompt:
                if isinstance(message, dict):
                    path = self._extract_audio_path_from_message(message)
                    if path is not None:
                        return path
            raise ValueError("Prompt list does not contain audio path in messages[*].audio(s).path")

        # Native NeMo manifest / full datapoint dict
        if isinstance(prompt, dict):
            if "messages" in prompt and isinstance(prompt["messages"], list):
                return self._extract_audio_reference(prompt["messages"])

            if "audio_filepath" in prompt and isinstance(prompt["audio_filepath"], str):
                return prompt["audio_filepath"]
            if "audio_filename" in prompt and isinstance(prompt["audio_filename"], str):
                return prompt["audio_filename"]
            if "audio_file" in prompt and isinstance(prompt["audio_file"], str):
                return prompt["audio_file"]
            if "context" in prompt and isinstance(prompt["context"], str):
                return prompt["context"]
            if "audio_path" in prompt:
                audio_path = prompt["audio_path"]
                if isinstance(audio_path, str):
                    return audio_path
                if isinstance(audio_path, list) and len(audio_path) > 0 and isinstance(audio_path[0], str):
                    return audio_path[0]

            path = self._extract_audio_path_from_message(prompt)
            if path is not None:
                return path

            raise ValueError(
                "Prompt dict does not contain supported audio key. Expected one of: "
                "messages[*].audio(s).path, audio_filepath, audio_filename, audio_file, context, or audio_path."
            )

        raise TypeError(f"Unsupported prompt type for NemoASRModel: {type(prompt)}")

    async def generate_async(self, prompt, **kwargs):
        """Transcribe audio file asynchronously.

        Args:
            prompt: Audio reference as path string, OpenAI-style messages, or manifest-like dict.
            **kwargs: Generation parameters (most LLM parameters are ignored, use extra_body for ASR options)

        Returns:
            dict: Result with 'generation' key containing transcription data
        """
        # Validate and warn about unsupported LLM parameters
        validate_unsupported_params(kwargs, "NemoASRModel")

        # Parse extra_body for ASR-specific options
        extra_body = kwargs.get("extra_body", {})
        language = extra_body.get("language_code", self.language_code)
        response_format = extra_body.get("response_format", self.response_format)
        enable_timestamps = extra_body.get("enable_timestamps", self.enable_timestamps)

        audio_path, cleanup_path = self._resolve_audio_from_prompt(prompt)
        try:
            # Check if chunking is needed
            chunk_duration = None
            if self.enable_audio_chunking:
                chunk_duration = await self._check_audio_duration(audio_path)

            # Prepare request
            start_time = time.time()

            # Read audio file
            async with aiofiles.open(audio_path, "rb") as f:
                audio_bytes = await f.read()

            # Prepare multipart form data
            files = {"file": (audio_path.name, audio_bytes, "audio/wav")}

            data = {
                "model": self.model_name_or_path,
                "language": language,
                "response_format": response_format,
            }

            # Add timestamp granularities if enabled
            if enable_timestamps and response_format == "verbose_json":
                data["timestamp_granularities"] = "word,segment"

            # Add chunking if needed
            if chunk_duration is not None:
                data["chunk_duration_sec"] = chunk_duration

            # Make request to server
            response = await self._client.post(f"{self.base_url}/v1/audio/transcriptions", files=files, data=data)
            response.raise_for_status()

            result_data = response.json()
            generation_time = time.time() - start_time

            # Extract transcription text
            pred_text = result_data["text"]

            # Build result in expected format
            result: Dict[str, Any] = {
                "pred_text": pred_text,
                "generation_time": generation_time,
                "audio_file": str(audio_path),
            }

            # Add words if available
            if "words" in result_data:
                result["words"] = result_data["words"]

            # Add any additional metadata
            if "language" in result_data:
                result["language"] = result_data["language"]
            if "duration" in result_data:
                result["duration"] = result_data["duration"]

            return {"generation": result}

        except httpx.HTTPStatusError as e:
            LOG.error(f"HTTP error during transcription: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"ASR server error: {e.response.text}") from e
        except Exception as e:
            LOG.error(f"ASR generation failed: {e}")
            raise
        finally:
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()

    async def _check_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Check audio duration and return chunk size if needed.

        Args:
            audio_path: Path to audio file

        Returns:
            Chunk duration in seconds if chunking is needed, None otherwise
        """
        try:
            import soundfile as sf
        except ImportError:
            LOG.warning("soundfile not available, skipping audio duration check")
            return None

        try:
            info = sf.info(str(audio_path))
            duration = info.duration

            if duration > self.chunk_audio_threshold_sec:
                LOG.info(
                    f"Audio duration ({duration:.1f}s) exceeds threshold "
                    f"({self.chunk_audio_threshold_sec}s), enabling chunking"
                )
                return self.chunk_audio_threshold_sec

        except Exception as e:
            LOG.warning(f"Failed to check audio duration: {e}")

        return None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._client.aclose()

    def __del__(self):
        """Clean up resources."""
        # Close SSH tunnel
        if hasattr(self, "_tunnel") and self._tunnel:
            try:
                self._tunnel.stop()
            except Exception:
                pass  # Ignore errors during cleanup
