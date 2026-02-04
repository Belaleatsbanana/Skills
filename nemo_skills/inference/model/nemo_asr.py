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

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    data_dir : str, default ""
        Base directory for resolving relative audio paths.
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
        data_dir: str = "",
        max_workers: int = 64,
        **kwargs,
    ) -> None:
        """Initialize NemoASRModel client."""
        
        # Handle base_url compatibility
        if base_url:
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
        
        # ASR-specific config
        self.language_code = language_code
        self.response_format = response_format
        self.enable_timestamps = enable_timestamps
        self.enable_audio_chunking = enable_audio_chunking
        self.chunk_audio_threshold_sec = chunk_audio_threshold_sec

        # Build base URL
        self.base_url = f"http://{host}:{port}"
        
        # Create HTTP client with connection pooling
        limits = httpx.Limits(max_keepalive_connections=max_workers, max_connections=max_workers)
        self._client = httpx.AsyncClient(limits=limits, timeout=300.0)  # 5 min timeout
        
        LOG.info(f"Initialized NemoASRModel connecting to {self.base_url}")

    async def generate_async(self, prompt: str, **kwargs):
        """Transcribe audio file asynchronously.

        Args:
            prompt: Path to audio file to transcribe (absolute or relative to data_dir)
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

        # Resolve audio file path
        audio_path = Path(prompt)
        if not audio_path.is_absolute():
            audio_path = Path(self.data_dir) / audio_path
        
        audio_path = audio_path.expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check if chunking is needed
        chunk_duration = None
        if self.enable_audio_chunking:
            chunk_duration = await self._check_audio_duration(audio_path)

        # Prepare request
        start_time = time.time()
        
        try:
            # Read audio file
            async with aiofiles.open(audio_path, 'rb') as f:
                audio_bytes = await f.read()

            # Prepare multipart form data
            files = {
                'file': (audio_path.name, audio_bytes, 'audio/wav')
            }
            
            data = {
                'model': self.model_name_or_path,
                'language': language,
                'response_format': response_format,
            }
            
            # Add timestamp granularities if enabled
            if enable_timestamps and response_format == "verbose_json":
                data['timestamp_granularities'] = 'word,segment'

            # Add chunking if needed
            if chunk_duration is not None:
                data['chunk_duration_sec'] = chunk_duration

            # Make request to server
            response = await self._client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data
            )
            response.raise_for_status()
            
            result_data = response.json()
            generation_time = time.time() - start_time

            # Extract transcription text
            pred_text = result_data.get("text", "")
            
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
        # Close HTTP client
        if hasattr(self, "_client"):
            try:
                # Try to close gracefully if event loop is available
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._client.aclose())
                else:
                    loop.run_until_complete(self._client.aclose())
            except Exception:
                pass  # Ignore errors during cleanup

        # Close SSH tunnel
        if hasattr(self, "_tunnel") and self._tunnel:
            try:
                self._tunnel.stop()
            except Exception:
                pass  # Ignore errors during cleanup
