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

"""NeMo ASR server with OpenAI-compatible API.

This server provides ASR inference using NeMo models (Canary, Parakeet, FastConformer)
with an OpenAI-compatible /v1/audio/transcriptions endpoint.
"""

import argparse
import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

LOG = logging.getLogger(__name__)


class NemoASRServer:
    """NeMo ASR server handling model loading and inference."""

    def __init__(self, model_path: str, num_gpus: int = 1):
        """Initialize NeMo ASR server.

        Args:
            model_path: Path to .nemo checkpoint or NGC model name (e.g., 'nvidia/canary-1b')
            num_gpus: Number of GPUs to use (currently only 1 is supported)
        """
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load NeMo ASR model from checkpoint or NGC."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo toolkit is not installed. Please install it with: "
                "pip install nemo_toolkit[asr]"
            )

        LOG.info(f"Loading NeMo ASR model from: {self.model_path}")
        start_time = time.time()

        # Check if it's a local .nemo file or NGC model name
        if os.path.exists(self.model_path) and self.model_path.endswith('.nemo'):
            LOG.info("Loading from local .nemo checkpoint")
            self.model = nemo_asr.models.ASRModel.restore_from(self.model_path)
        else:
            LOG.info("Loading from NGC or model name")
            try:
                self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_path)
            except Exception as e:
                LOG.error(f"Failed to load model from NGC: {e}")
                LOG.info("Attempting to load as local path...")
                self.model = nemo_asr.models.ASRModel.restore_from(self.model_path)

        # Move model to GPU if available
        if self.num_gpus > 0:
            import torch
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                LOG.info(f"Model moved to GPU")

        self.model.eval()
        load_time = time.time() - start_time
        LOG.info(f"Model loaded successfully in {load_time:.2f}s")

    async def _transcribe_with_chunking(
        self,
        audio_path: str,
        chunk_duration_sec: float,
        enable_timestamps: bool = False,
    ) -> tuple[str, float]:
        """Transcribe long audio by chunking it into smaller segments.

        Args:
            audio_path: Path to audio file
            chunk_duration_sec: Duration of each chunk in seconds
            enable_timestamps: Whether to enable timestamps

        Returns:
            Tuple of (transcribed_text, total_inference_time)
        """
        try:
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise ImportError("soundfile and numpy are required for audio chunking")

        # Load audio file
        audio_array, sampling_rate = sf.read(audio_path)
        duration = len(audio_array) / sampling_rate
        
        LOG.info(f"Chunking audio ({duration:.1f}s) into segments of {chunk_duration_sec}s")

        # Calculate chunks
        chunk_samples = int(chunk_duration_sec * sampling_rate)
        num_chunks = int(np.ceil(len(audio_array) / chunk_samples))

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, len(audio_array))
            chunk = audio_array[start:end]
            
            # Merge tiny trailing chunks
            min_chunk_samples = int(0.5 * sampling_rate)  # 0.5 second minimum
            if len(chunk) < min_chunk_samples and chunks:
                chunks[-1] = np.concatenate([chunks[-1], chunk])
            else:
                chunks.append(chunk)

        LOG.info(f"Created {len(chunks)} audio chunks")

        # Transcribe each chunk
        chunk_texts = []
        total_time = 0.0

        for chunk_idx, audio_chunk in enumerate(chunks):
            # Save chunk to temporary file
            chunk_path = f"{audio_path}.chunk_{chunk_idx}.wav"
            try:
                sf.write(chunk_path, audio_chunk, sampling_rate)

                # Transcribe chunk
                start_time = time.time()
                hypotheses = self.model.transcribe(
                    [chunk_path],
                    batch_size=1,
                    return_hypotheses=True,
                    timestamps=enable_timestamps
                )
                chunk_time = time.time() - start_time
                total_time += chunk_time

                if len(hypotheses) > 0:
                    text = hypotheses[0][0].text
                    chunk_texts.append(text.strip())
                    LOG.debug(f"Chunk {chunk_idx + 1}/{len(chunks)}: {text[:50]}...")

            finally:
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)

        # Concatenate all chunk transcriptions
        full_text = " ".join(chunk_texts)
        
        return full_text, total_time

    async def transcribe(
        self,
        audio_file: UploadFile,
        language: Optional[str] = None,
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None,
        chunk_duration_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio file using NeMo ASR model.

        Args:
            audio_file: Audio file to transcribe
            language: Language code (e.g., 'en', 'es') - optional
            response_format: 'json' or 'verbose_json'
            timestamp_granularities: List of granularities ['word', 'segment']
            chunk_duration_sec: If specified, chunk audio into segments of this duration

        Returns:
            Transcription result in OpenAI-compatible format
        """
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Determine if timestamps are needed
            enable_timestamps = (
                timestamp_granularities is not None 
                and len(timestamp_granularities) > 0
                and response_format == "verbose_json"
            )

            # Handle chunking if requested
            if chunk_duration_sec is not None and chunk_duration_sec > 0:
                text, inference_time = await self._transcribe_with_chunking(
                    tmp_path, chunk_duration_sec, enable_timestamps
                )
            else:
                # Transcribe using NeMo
                start_time = time.time()
                hypotheses = self.model.transcribe(
                    [tmp_path],
                    batch_size=1,
                    return_hypotheses=True,
                    timestamps=enable_timestamps
                )
                inference_time = time.time() - start_time

                # Extract transcription
                if len(hypotheses) == 0:
                    raise RuntimeError("No transcription returned from model")

                hypothesis = hypotheses[0][0]  # [batch_idx][hypothesis_idx]
                text = hypothesis.text

            # Build response based on format
            result = {"text": text}

            if response_format == "verbose_json":
                result["task"] = "transcribe"
                result["duration"] = None  # Could compute from audio file if needed
                
                # Add language if detected/specified
                if language:
                    result["language"] = language

                # Add timestamps if requested
                if enable_timestamps and hasattr(hypothesis, 'timestep'):
                    words = []
                    
                    # Extract word-level timestamps
                    if 'word' in timestamp_granularities:
                        word_timestamps = getattr(hypothesis.timestep, 'word', [])
                        for word_info in word_timestamps:
                            words.append({
                                "word": word_info['word'],
                                "start": word_info['start_offset'],
                                "end": word_info['end_offset']
                            })
                    
                    if words:
                        result["words"] = words

                # Add inference metadata
                result["inference_time"] = inference_time

            return result

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def create_app(model_path: str, num_gpus: int = 1) -> FastAPI:
    """Create FastAPI application with NeMo ASR server.

    Args:
        model_path: Path to model or NGC name
        num_gpus: Number of GPUs to use

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="NeMo ASR Server",
        description="OpenAI-compatible ASR server using NeMo models",
        version="1.0.0"
    )

    # Initialize server
    server = NemoASRServer(model_path, num_gpus)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "model": model_path}

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(..., description="Audio file to transcribe"),
        model: str = Form(default="nemo-asr", description="Model to use (ignored, using server model)"),
        language: Optional[str] = Form(default=None, description="Language code"),
        response_format: str = Form(default="json", description="Response format: json or verbose_json"),
        timestamp_granularities: Optional[str] = Form(
            default=None, 
            description="Comma-separated list: word,segment"
        ),
        chunk_duration_sec: Optional[float] = Form(
            default=None,
            description="If specified, chunk audio into segments of this duration (in seconds)"
        ),
    ):
        """Transcribe audio file.

        OpenAI-compatible endpoint for audio transcription.
        """
        try:
            # Parse timestamp granularities
            granularities = None
            if timestamp_granularities:
                granularities = [g.strip() for g in timestamp_granularities.split(',')]

            result = await server.transcribe(
                audio_file=file,
                language=language,
                response_format=response_format,
                timestamp_granularities=granularities,
                chunk_duration_sec=chunk_duration_sec
            )

            return JSONResponse(content=result)

        except Exception as e:
            LOG.error(f"Transcription failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    """Main entry point for NeMo ASR server."""
    parser = argparse.ArgumentParser(description="Serve NeMo ASR model")
    parser.add_argument("--model", required=True, help="Path to model or NGC model name")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes (not used, for compatibility)")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run app
    app = create_app(args.model, args.num_gpus)
    
    LOG.info(f"Starting NeMo ASR server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
