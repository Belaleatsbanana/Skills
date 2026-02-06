#!/usr/bin/env python3
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

"""
CLI wrapper for the Unified NeMo Inference Server.

This module provides a command-line interface compatible with nemo-skills
server deployment patterns. It translates standard vllm-style CLI arguments
to the unified server configuration.

Usage via NeMo-Skills:

    # SALM backend (speech-augmented language model)
    ns eval \\
        --server_type vllm \\
        --server_gpus 1 \\
        --model /path/to/model \\
        --server_entrypoint "-m nemo_skills.inference.server.serve_unified" \\
        --server_args "--backend salm"

    # MagpieTTS backend (text-to-speech with RTF metrics)
    ns eval \\
        --server_type vllm \\
        --server_gpus 1 \\
        --model /path/to/tts_model \\
        --server_entrypoint "-m nemo_skills.inference.server.serve_unified" \\
        --server_args "--backend magpie_tts --codec_model /path/to/codec"

    # S2S backend (speech-to-speech)
    ns eval \\
        --server_type vllm \\
        --server_gpus 1 \\
        --model /path/to/s2s_model \\
        --server_entrypoint "-m nemo_skills.inference.server.serve_unified" \\
        --server_args "--backend s2s"

Environment Variables:
    UNIFIED_SERVER_HOST: Server host (default: 0.0.0.0)
    UNIFIED_SERVER_PORT: Server port (default: 8000)
    UNIFIED_SERVER_BACKEND: Backend type (default: salm)
    UNIFIED_SERVER_MODEL_PATH: Path to model
    UNIFIED_SERVER_CODEC_MODEL_PATH: Path to codec model
    UNIFIED_SERVER_BATCH_SIZE: Batch size (default: 8)
    UNIFIED_SERVER_BATCH_TIMEOUT: Batch timeout (default: 0.1)
    DEBUG: Enable debug mode
"""

import argparse
import inspect
import os
import shutil
import sys
from typing import Optional


def setup_pythonpath(code_path: Optional[str] = None):
    """Set up PYTHONPATH for NeMo and the unified server.

    Args:
        code_path: Single path or colon-separated paths to add to PYTHONPATH
    """
    paths_to_add = []

    # Add explicit code path(s) if provided (supports colon-separated paths)
    if code_path:
        for path in code_path.split(":"):
            if path and path not in paths_to_add:
                paths_to_add.append(path)

    # Add recipes path for unified server imports
    # Look for the recipes directory relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to find ns_eval root (go up from nemo_skills/inference/server/)
    ns_eval_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
    if os.path.exists(os.path.join(ns_eval_root, "recipes")):
        paths_to_add.append(ns_eval_root)

    # Also check /nemo_run/code pattern used in containers
    if os.path.exists("/nemo_run/code"):
        paths_to_add.append("/nemo_run/code")

    # Update PYTHONPATH
    current_path = os.environ.get("PYTHONPATH", "")
    for path in paths_to_add:
        if path not in current_path.split(":"):
            current_path = f"{path}:{current_path}" if current_path else path

    os.environ["PYTHONPATH"] = current_path

    # Also add to sys.path for immediate imports
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)


def apply_safetensors_patch(hack_path: Optional[str]):
    """Apply safetensors patch if provided (for some NeMo models)."""
    if not hack_path or not os.path.exists(hack_path):
        return

    try:
        import safetensors.torch as st_torch

        dest_path = inspect.getfile(st_torch)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copyfile(hack_path, dest_path)
        print(f"[serve_unified] Applied safetensors patch: {hack_path} -> {dest_path}")
    except Exception as e:
        print(f"[serve_unified] Warning: Failed to apply safetensors patch: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified NeMo Inference Server CLI wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Standard vllm-style arguments (for nemo-skills compatibility)
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    # Backend selection
    parser.add_argument(
        "--backend",
        default="salm",
        choices=["salm", "magpie_tts", "ear_tts", "ear_tts_batch", "s2s", "s2s_incremental", "s2s_session"],
        help="Backend type: salm (speech-augmented LM), magpie_tts (MagpieTTS with RTF metrics), ear_tts (EAR TTS streaming decode), ear_tts_batch (EAR TTS batch decode), s2s (speech-to-speech offline), s2s_incremental (frame-by-frame processing), s2s_session (session-aware multi-turn)",
    )

    # Backend-specific model paths
    parser.add_argument("--codec_model", default=None, help="Path to codec model (required for TTS, optional for S2S)")

    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--batch_size", type=int, default=8, help="Maximum batch size")
    parser.add_argument(
        "--batch_timeout", type=float, default=0.1, help="Batch timeout in seconds (0 for no batching delay)"
    )

    # Generation defaults
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")

    # Model configuration
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")

    # Backend-specific options
    parser.add_argument("--prompt_format", default=None, help="Prompt format (SALM backend)")
    parser.add_argument(
        "--phoneme_input_type", default="predicted", help="Phoneme input type: predicted or gt (TTS backend)"
    )
    parser.add_argument(
        "--decoder_only_model", action="store_true", help="Use decoder-only model architecture (TTS backend)"
    )
    parser.add_argument("--use_local_transformer", action="store_true", help="Use local transformer (TTS backend)")
    parser.add_argument("--top_k", type=int, default=80, help="Top-k sampling (TTS backend)")
    parser.add_argument("--use_cfg", action="store_true", help="Enable classifier-free guidance (TTS backend)")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="CFG scale factor (TTS backend)")
    parser.add_argument("--save_codes", action="store_true", help="Save predicted codec codes for FCD scoring (TTS backend)")

    # Checkpoint loading options (for magpie_tts backend - alternative to --model .nemo)
    parser.add_argument("--hparams_file", default=None, help="Path to hparams.yaml (use with --checkpoint_file)")
    parser.add_argument("--checkpoint_file", default=None, help="Path to .ckpt checkpoint (use with --hparams_file)")
    parser.add_argument(
        "--legacy_codebooks", action="store_true", help="Use legacy codebook indices for old checkpoints"
    )
    parser.add_argument("--legacy_text_conditioning", action="store_true", help="Use legacy text conditioning")
    parser.add_argument("--hparams_from_wandb", action="store_true", help="hparams file was exported from wandb")

    # Environment setup
    parser.add_argument("--code_path", default=None, help="Path to NeMo source code to add to PYTHONPATH")
    parser.add_argument("--hack_path", default=None, help="Path to safetensors/torch.py patch file")

    # S2S backend options
    parser.add_argument(
        "--ignore_system_prompt",
        action="store_true",
        help="Ignore system prompts from requests (for models that don't support them)",
    )
    parser.add_argument(
        "--silence_padding_sec",
        type=float,
        default=5.0,
        help="Seconds of silence to append after audio (S2S backends)",
    )

    # S2S Incremental backend options
    parser.add_argument(
        "--config_path",
        default=None,
        help="Path to YAML config file (s2s_incremental backend)",
    )
    parser.add_argument(
        "--llm_checkpoint_path",
        default=None,
        help="Path to LLM checkpoint (s2s_incremental backend)",
    )
    parser.add_argument(
        "--tts_checkpoint_path",
        default=None,
        help="Path to TTS checkpoint (s2s_incremental backend)",
    )
    parser.add_argument(
        "--speaker_reference",
        default=None,
        help="Path to speaker reference audio for TTS (s2s_incremental backend)",
    )
    parser.add_argument(
        "--num_frames_per_inference",
        type=int,
        default=1,
        help="Frames per inference step (s2s_incremental backend)",
    )
    parser.add_argument(
        "--no_decode_audio",
        action="store_true",
        help="Disable audio output (s2s_incremental backend)",
    )

    # Session management options (s2s_session backend)
    parser.add_argument(
        "--session_ttl",
        type=float,
        default=300.0,
        help="Session time-to-live in seconds (s2s_session backend)",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=100,
        help="Maximum number of concurrent sessions (s2s_session backend)",
    )
    parser.add_argument(
        "--session_artifacts_dir",
        type=str,
        default=None,
        help="Directory to save session artifacts (input/output audio, JSON). Default: /tmp/s2s_sessions",
    )
    parser.add_argument(
        "--no_save_session_artifacts",
        action="store_true",
        help="Disable saving session artifacts to disk",
    )
    parser.add_argument(
        "--output_frame_alignment",
        action="store_true",
        help="Include per-frame alignment data in debug output (user/agent/ASR per frame)",
    )

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Parse known args, allowing extra args to be passed through
    args, extra_args = parser.parse_known_args()

    # Setup environment
    setup_pythonpath(args.code_path)
    apply_safetensors_patch(args.hack_path)

    # Store code_path for backends that may need to add paths late
    if args.code_path:
        os.environ["UNIFIED_SERVER_CODE_PATH"] = args.code_path

    # Set environment variables
    os.environ["UNIFIED_SERVER_HOST"] = args.host
    os.environ["UNIFIED_SERVER_PORT"] = str(args.port)
    os.environ["UNIFIED_SERVER_BACKEND"] = args.backend
    os.environ["UNIFIED_SERVER_MODEL_PATH"] = args.model
    os.environ["UNIFIED_SERVER_BATCH_SIZE"] = str(args.batch_size)
    os.environ["UNIFIED_SERVER_BATCH_TIMEOUT"] = str(args.batch_timeout)
    os.environ["UNIFIED_SERVER_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["UNIFIED_SERVER_TEMPERATURE"] = str(args.temperature)
    os.environ["UNIFIED_SERVER_TOP_P"] = str(args.top_p)

    if args.codec_model:
        os.environ["UNIFIED_SERVER_CODEC_MODEL_PATH"] = args.codec_model

    if args.debug:
        os.environ["DEBUG"] = "1"

    # Set CUDA devices (only if not already set by the environment, e.g., SLURM)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))

    # Build extra config for backend-specific options
    extra_config = {}

    if args.prompt_format:
        extra_config["prompt_format"] = args.prompt_format

    if args.backend == "magpie_tts":
        extra_config["decoder_only_model"] = args.decoder_only_model
        extra_config["phoneme_input_type"] = args.phoneme_input_type
        extra_config["use_local_transformer"] = args.use_local_transformer
        extra_config["top_k"] = args.top_k
        extra_config["use_cfg"] = args.use_cfg
        extra_config["cfg_scale"] = args.cfg_scale
        extra_config["save_codes"] = args.save_codes
        # Checkpoint loading options
        if args.hparams_file:
            extra_config["hparams_file"] = args.hparams_file
        if args.checkpoint_file:
            extra_config["checkpoint_file"] = args.checkpoint_file
        if args.legacy_codebooks:
            extra_config["legacy_codebooks"] = True
        if args.legacy_text_conditioning:
            extra_config["legacy_text_conditioning"] = True
        if args.hparams_from_wandb:
            extra_config["hparams_from_wandb"] = True

    # S2S backend options
    if args.backend in ("s2s", "s2s_incremental", "s2s_session"):
        extra_config["ignore_system_prompt"] = args.ignore_system_prompt
        if args.silence_padding_sec != 5.0:
            extra_config["silence_padding_sec"] = args.silence_padding_sec

    # EAR TTS backend options
    if args.backend in ("ear_tts", "ear_tts_batch"):
        if args.config_path:
            extra_config["config_path"] = args.config_path
        if args.tts_checkpoint_path:
            extra_config["tts_checkpoint_path"] = args.tts_checkpoint_path
        if args.speaker_reference:
            extra_config["speaker_reference"] = args.speaker_reference

    # S2S Incremental/Session backend options (shared config)
    if args.backend in ("s2s_incremental", "s2s_session"):
        if args.config_path:
            extra_config["config_path"] = args.config_path
        if args.llm_checkpoint_path:
            extra_config["llm_checkpoint_path"] = args.llm_checkpoint_path
        if args.tts_checkpoint_path:
            extra_config["tts_checkpoint_path"] = args.tts_checkpoint_path
        if args.speaker_reference:
            extra_config["speaker_reference"] = args.speaker_reference
        if args.num_frames_per_inference != 1:
            extra_config["num_frames_per_inference"] = args.num_frames_per_inference
        if args.no_decode_audio:
            extra_config["decode_audio"] = False
        # Artifacts and alignment (available for both backends)
        if args.session_artifacts_dir:
            extra_config["session_artifacts_dir"] = args.session_artifacts_dir
        extra_config["save_session_artifacts"] = not args.no_save_session_artifacts
        extra_config["output_frame_alignment"] = args.output_frame_alignment

    # S2S Session backend options
    if args.backend == "s2s_session":
        extra_config["session_ttl"] = args.session_ttl
        extra_config["max_sessions"] = args.max_sessions

    # Print configuration
    print("=" * 60)
    print("[serve_unified] Starting Unified NeMo Inference Server")
    print("=" * 60)
    print(f"  Backend: {args.backend}")
    print(f"  Model: {args.model}")
    if args.codec_model:
        print(f"  Codec Model: {args.codec_model}")
    print(f"  Port: {args.port}")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Batch Timeout: {args.batch_timeout}s")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    if args.backend == "magpie_tts":
        print(f"  Top-k: {args.top_k}")
        print(f"  CFG: {args.use_cfg} (scale: {args.cfg_scale})")
        print(f"  Local Transformer: {args.use_local_transformer}")
        if args.hparams_file and args.checkpoint_file:
            print(f"  Hparams: {args.hparams_file}")
            print(f"  Checkpoint: {args.checkpoint_file}")
            if args.legacy_codebooks:
                print("  Legacy Codebooks: True")
            if args.legacy_text_conditioning:
                print("  Legacy Text Conditioning: True")
    if args.backend in ("ear_tts", "ear_tts_batch"):
        if args.config_path:
            print(f"  Config Path: {args.config_path}")
        if args.speaker_reference:
            print(f"  Speaker Reference: {args.speaker_reference}")
        if args.tts_checkpoint_path:
            print(f"  TTS Checkpoint: {args.tts_checkpoint_path}")
    if args.backend in ("s2s_incremental", "s2s_session"):
        if args.config_path:
            print(f"  Config Path: {args.config_path}")
        if args.llm_checkpoint_path:
            print(f"  LLM Checkpoint: {args.llm_checkpoint_path}")
        if args.speaker_reference:
            print(f"  Speaker Reference: {args.speaker_reference}")
        print(f"  Frames per Inference: {args.num_frames_per_inference}")
        print(f"  Decode Audio: {not args.no_decode_audio}")
        print(f"  Save Artifacts: {not args.no_save_session_artifacts}")
        if args.session_artifacts_dir:
            print(f"  Artifacts Dir: {args.session_artifacts_dir}")
        else:
            print("  Artifacts Dir: /tmp/s2s_sessions (default)")
        print(f"  Output Frame Alignment: {args.output_frame_alignment}")
    if args.backend == "s2s_session":
        print(f"  Session TTL: {args.session_ttl}s")
        print(f"  Max Sessions: {args.max_sessions}")
    if extra_config:
        print(f"  Extra Config: {extra_config}")
    print("=" * 60)

    # Import and run the unified server
    try:
        import uvicorn

        from recipes.multimodal.server.unified_server import create_app

        app = create_app(
            backend_type=args.backend,
            model_path=args.model,
            codec_model_path=args.codec_model or "",
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            device=args.device,
            dtype=args.dtype,
            extra_config=extra_config if extra_config else None,
        )

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    except ImportError as e:
        print(f"[serve_unified] Error: Failed to import unified server: {e}")
        print("[serve_unified] Make sure the recipes.multimodal.server package is in PYTHONPATH")
        sys.exit(1)
    except Exception as e:
        print(f"[serve_unified] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
