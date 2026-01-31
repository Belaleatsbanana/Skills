#!/usr/bin/env python3
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

"""Prepare EmergentTTS-Eval benchmark for NeMo-Skills.

This script:
1) Downloads the EmergentTTS-Eval HF dataset
2) Saves baseline audios to wav files
3) Writes `data/emergent_tts_eval_data.jsonl` in Emergent's expected schema
4) Downloads `data/wv_mos.ckpt`
5) Writes NeMo-Skills `test.jsonl` for generation (OpenAI prompt format)

Typical usage (to create everything under your shared NeMo-Skills data dir):
  python prepare.py --output_dir /lustre/.../data_dir/emergent_tts
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from urllib.error import ContentTooShortError
from pathlib import Path


SYSTEM_MESSAGE = "You are a helpful assistant."
DEFAULT_DATASET = "bosonai/EmergentTTS-Eval"
DEFAULT_SPLIT = "train"
WV_MOS_URL = "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1"


def _require_deps():
    try:
        import numpy as np  # noqa: F401
        from datasets import load_dataset  # noqa: F401
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
        from pydub import AudioSegment  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies for EmergentTTS-Eval preparation.\n\n"
            "Install into the repo venv:\n"
            "  cd /home/vmendelev/workspace/expressiveness/src/nemo-skills-tts-eval\n"
            "  . ./.venv/bin/activate\n"
            "  pip install datasets numpy pydub tqdm librosa soundfile\n"
        ) from e


def _download_wv_mos(dst_path: Path, overwrite: bool) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and not overwrite:
        return
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")

    # Zenodo downloads can occasionally fail with ContentTooShortError; retry.
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            urllib.request.urlretrieve(WV_MOS_URL, str(tmp_path))
            tmp_path.replace(dst_path)
            return
        except ContentTooShortError as e:
            # Partial download: wait and retry.
            wait_s = min(5 * attempt, 30)
            print(f"Warning: partial download for wv_mos.ckpt (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(wait_s)
        except Exception as e:
            wait_s = min(5 * attempt, 30)
            print(f"Warning: failed downloading wv_mos.ckpt (attempt {attempt}/{max_attempts}): {e}")
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to download wv_mos.ckpt after {max_attempts} attempts: {WV_MOS_URL}")


def _write_benchmark_init(bench_dir: Path) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    init_path = bench_dir / "__init__.py"
    init_path.write_text(
        (
            "# EmergentTTS-Eval benchmark (NeMo-Skills)\n\n"
            'GENERATION_ARGS = "++prompt_format=openai"\n'
        ),
        encoding="utf-8",
    )


def _to_nemo_skills_entry(sample: dict) -> dict:
    # MagpieTTS backend expects JSON with at least `text`. We also keep Emergent
    # metadata to enable deterministic conversion/scoring later.
    payload = {
        "text": sample["text_to_synthesize"],
        "text_to_synthesize": sample["text_to_synthesize"],
        "category": sample["category"],
        "evolution_depth": sample["evolution_depth"],
        "language": sample["language"],
        "unique_id_eval": sample["unique_id_eval"],
        # Optional fields used by MagpieTTS evaluation code paths.
        "context_audio_filepath": "",
        "duration": 5.0,
        "context_audio_duration": 5.0,
    }
    return {
        "problem": "",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }


def main():
    _require_deps()
    import numpy as np
    from datasets import load_dataset
    from pydub import AudioSegment
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Prepare EmergentTTS-Eval for NeMo-Skills")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Where to create emergent_tts module structure (default: folder containing this script).",
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="HF dataset name")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="HF split to download (train contains 1645)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (baseline audios, jsonl, wv_mos.ckpt, test.jsonl).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Optional: limit number of samples (debug). If set, takes the first N rows.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    data_dir = output_dir / "data"
    baseline_audios_dir = data_dir / "baseline_audios"
    baseline_audios_dir.mkdir(parents=True, exist_ok=True)

    # Emergent expected files
    emergent_jsonl_path = data_dir / "emergent_tts_eval_data.jsonl"
    wv_mos_path = data_dir / "wv_mos.ckpt"

    # NeMo-Skills benchmark module structure
    bench_dir = output_dir / "emergent"
    test_jsonl_path = bench_dir / "test.jsonl"
    _write_benchmark_init(bench_dir)

    # Download dataset
    dataset_hf = load_dataset(args.dataset, split=args.split)
    total = len(dataset_hf) if args.num_samples is None else min(args.num_samples, len(dataset_hf))

    if emergent_jsonl_path.exists() and test_jsonl_path.exists() and not args.overwrite:
        print(f"Found existing outputs under {output_dir}. Use --overwrite to rebuild.")
    else:
        if args.overwrite:
            for p in [emergent_jsonl_path, test_jsonl_path]:
                if p.exists():
                    p.unlink()

        emergent_records: list[dict] = []

        # Build emergent jsonl + baseline audios
        for i in tqdm(range(total), desc="Preparing EmergentTTS-Eval"):
            curr = dataset_hf[i]
            unique_id = i

            # Save baseline audio
            wav_path = baseline_audios_dir / f"{unique_id}.wav"
            if args.overwrite or not wav_path.exists():
                audio_array = curr["audio"]["array"]
                audio_sr = int(curr["audio"]["sampling_rate"])
                audio_array_int16 = np.int16(audio_array * 32767)
                audio_segment = AudioSegment(
                    audio_array_int16.tobytes(),
                    frame_rate=audio_sr,
                    sample_width=2,
                    channels=1,
                )
                audio_segment.export(str(wav_path), format="wav")

            emergent_records.append(
                {
                    "unique_id_eval": unique_id,
                    "category": curr["category"],
                    "text_to_synthesize": curr["text_to_synthesize"],
                    "evolution_depth": curr["evolution_depth"],
                    "language": curr["language"],
                }
            )

        # Write emergent jsonl data file
        emergent_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(emergent_jsonl_path, "w", encoding="utf-8") as f:
            for rec in emergent_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write NeMo-Skills test.jsonl
        with open(test_jsonl_path, "w", encoding="utf-8") as f:
            for rec in emergent_records:
                f.write(json.dumps(_to_nemo_skills_entry(rec), ensure_ascii=False) + "\n")

    # Download MOS model checkpoint (used by Emergent scoring)
    _download_wv_mos(wv_mos_path, overwrite=args.overwrite)

    print("\nPrepared EmergentTTS-Eval:")
    print(f"  - data dir: {data_dir}")
    print(f"  - baseline audios: {baseline_audios_dir}")
    print(f"  - emergent jsonl: {emergent_jsonl_path}")
    print(f"  - wv_mos.ckpt: {wv_mos_path}")
    print(f"  - nemo-skills test.jsonl: {test_jsonl_path}")


if __name__ == "__main__":
    main()

