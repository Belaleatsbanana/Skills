# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Prepare eval-results for FDB scoring: copy audio to fdb_prepared, optionally run FDB ASR
to produce time-aligned output.json per sample.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from typing import Optional

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    from scipy.io import wavfile as scipy_wavfile
except ImportError:
    scipy_wavfile = None


def _version_folders(fdb_version: str):
    """Return (primary, alternate) folder names for the given FDB version (e.g. v1.0 -> ('v1.0', 'v1_0'))."""
    if fdb_version == "v1.5":
        return ("v1.5", "v1_5")
    return ("v1.0", "v1_0")


def _get_sample_id_and_folder(subtest: str, entry_id: str, fdb_version: str) -> Optional[tuple]:
    """Return (sample_id, fdb_folder_name) for looking up input.wav in fdb_data_path, or None."""
    entry_id = str(entry_id)
    if fdb_version == "v1.5":
        if entry_id.startswith("user_interruption_"):
            return (entry_id.replace("user_interruption_", ""), "user_interruption")
        # v1.5 turn_taking if we ever support it
        return None
    # v1.0
    if subtest == "turn_taking":
        if entry_id.startswith("candor_turn_taking_"):
            return (entry_id.replace("candor_turn_taking_", ""), "candor_turn_taking")
        if entry_id.isdigit():
            return (entry_id, "candor_turn_taking")
    if subtest == "interruption":
        if entry_id.startswith("synthetic_user_interruption_"):
            return (entry_id.replace("synthetic_user_interruption_", ""), "synthetic_user_interruption")
        if entry_id.isdigit():
            return (entry_id, "synthetic_user_interruption")
    return None


def _get_input_wav_path(
    subtest: str,
    entry_id: str,
    fdb_data_path: Path,
    fdb_version: str,
) -> Optional[Path]:
    """Return path to input.wav in FDB data for this sample, or None if not found."""
    t = _get_sample_id_and_folder(subtest, entry_id, fdb_version)
    if t is None:
        return None
    sample_id, folder = t
    vers = _version_folders(fdb_version)
    for ver in vers:
        candidate = fdb_data_path / ver / folder / sample_id / "input.wav"
        if candidate.exists():
            return candidate
    return None


def _ensure_wav_to_dest(src: Path, dest: Path, stereo: bool = False) -> None:
    """Copy wav to dest. For FDB (stereo=True) the source MUST already be 2-channel; we do not convert mono to stereo.
    Pipeline: server returns audio -> vllm_multimodal saves bytes to output_dir/audio/*.wav -> we copy to fdb_prepared/<id>/output.wav.
    Ensure the model/server is configured to output 2-channel audio so we never receive mono."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def is_stereo(data: np.ndarray) -> bool:
        return data.ndim == 2 and data.shape[1] == 2

    # Prefer soundfile (handles float and int)
    if sf is not None:
        try:
            data, sr = sf.read(str(src))
            if stereo:
                if not is_stereo(data):
                    raise ValueError(
                        f"FDB requires 2-channel (stereo) output. Got {data.ndim}D array with shape {getattr(data, 'shape', '?')} from {src}. "
                        "Configure the model/server (S2S-Duplex inference or serve_unified backend) to output stereo; do not convert mono to stereo here."
                    )
                # Already stereo: write as-is
            else:
                if data.ndim > 1:
                    data = data.mean(axis=1)
            sf.write(str(dest), data, sr)
            return
        except ValueError:
            raise
        except Exception:
            pass

    # Fallback: scipy.io.wavfile (int only)
    if scipy_wavfile is not None:
        try:
            sr, data = scipy_wavfile.read(str(src))
            if stereo:
                if not is_stereo(data):
                    raise ValueError(
                        f"FDB requires 2-channel (stereo) output. Got {data.ndim}D array with shape {getattr(data, 'shape', '?')} from {src}. "
                        "Configure the model/server to output stereo; do not convert mono to stereo here."
                    )
            else:
                if data.ndim > 1:
                    data = data.mean(axis=1)
            scipy_wavfile.write(str(dest), sr, data)
            return
        except ValueError:
            raise
        except Exception:
            pass

    # No reader: copy as-is (stereo requirement cannot be verified)
    if stereo:
        print(f"Warning: Could not read {src} with soundfile/scipy; copying as-is. Ensure the file is 2-channel for FDB.", file=sys.stderr)
    shutil.copy2(src, dest)


def prepare_fdb_dir(
    eval_results_dir: Path,
    output_jsonl: Path,
    fdb_repo: Path,
    subdir_name: str = "fdb_prepared",
    asr_task: Optional[str] = None,
    run_asr: bool = False,
    fdb_data_path: Optional[Path] = None,
    subtest: Optional[str] = None,
    fdb_version: str = "v1.0",
    stereo: bool = True,
) -> Path:
    """Build FDB-format dir under eval_results_dir/subdir_name. Returns path to prepared dir.
    output.wav must be 2-channel from the model/server; we do not convert mono to stereo (we verify and fail if mono)."""
    fdb_prepared = eval_results_dir / subdir_name
    audio_dir = eval_results_dir / "audio"
    entries_with_audio = []

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry_id = entry.get("id") or entry.get("sample_id") or f"sample_{len(entries_with_audio)}"
            audio_path = None
            if "audio" in entry and isinstance(entry["audio"], dict):
                audio_path = entry["audio"].get("path")
            if audio_path and os.path.isabs(audio_path) and os.path.exists(audio_path):
                entries_with_audio.append((entry_id, Path(audio_path)))
            elif audio_path:
                for base in [audio_dir, eval_results_dir]:
                    p = base / Path(audio_path).name if not str(audio_path).startswith("audio/") else base / audio_path
                    if p.exists():
                        entries_with_audio.append((entry_id, p))
                        break

    if not entries_with_audio:
        for wav in (audio_dir if audio_dir.exists() else eval_results_dir).rglob("*.wav"):
            entry_id = wav.stem
            entries_with_audio.append((entry_id, wav))

    if not entries_with_audio:
        print("No audio found in output.jsonl or audio/")
        return fdb_prepared

    def fdb_dir_name(entry_id: str) -> str:
        """FDB backchannel eval only considers dirs whose name is purely numeric (spk.isdigit())."""
        if subtest == "backchannel":
            e = str(entry_id)
            if e.startswith("icc_backchannel_"):
                return e.replace("icc_backchannel_", "", 1)
            if e.startswith("user_backchannel_"):  # v1.5
                return e.replace("user_backchannel_", "", 1)
            if e.isdigit():
                return e
        return str(entry_id)

    fdb_prepared.mkdir(parents=True, exist_ok=True)
    # Match reference (reorganize_candor_outputs): output.wav = model response only; copy input.wav from original for turn_taking/interruption so dir has both.
    copy_input_wav_to_dir = subtest in ("turn_taking", "interruption")
    for entry_id, src_wav in entries_with_audio:
        dest_dir = fdb_prepared / fdb_dir_name(entry_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_wav = dest_dir / "output.wav"
        _ensure_wav_to_dest(src_wav, dest_wav, stereo=stereo)
        if copy_input_wav_to_dir and fdb_data_path and fdb_data_path.exists():
            input_wav_path = _get_input_wav_path(subtest, entry_id, fdb_data_path, fdb_version)
            if input_wav_path is not None:
                shutil.copy2(input_wav_path, dest_dir / "input.wav")

    # FDB turn_taking eval requires turn_taking.json (input turn end time) in each sample dir
    vers = _version_folders(fdb_version)
    if subtest == "turn_taking" and fdb_data_path and fdb_data_path.exists():
        for entry_id, _ in entries_with_audio:
            entry_id = str(entry_id)
            if entry_id.startswith("candor_turn_taking_"):
                sample_id = entry_id.replace("candor_turn_taking_", "")
            elif entry_id.isdigit():
                sample_id = entry_id  # e.g. id "1" -> candor_turn_taking/1/
            else:
                continue
            for ver in vers:
                src = fdb_data_path / ver / "candor_turn_taking" / sample_id / "turn_taking.json"
                if src.exists():
                    shutil.copy2(src, fdb_prepared / entry_id / "turn_taking.json")
                    break
            else:
                print(f"Warning: turn_taking.json not found for {entry_id} (tried {vers})")

    # FDB ASR for interruption expects interrupt.json in each sample dir (to crop audio after interrupt)
    # v1.0 uses folder "synthetic_user_interruption" and has interrupt.json; v1.5 uses "user_interruption" and has metadata.json
    if subtest == "interruption" and fdb_data_path and fdb_data_path.exists():
        folder_v1_0 = "synthetic_user_interruption"
        folder_v1_5 = "user_interruption"
        for entry_id, _ in entries_with_audio:
            entry_id = str(entry_id)
            sample_id = None
            if entry_id.startswith("synthetic_user_interruption_"):
                sample_id = entry_id.replace("synthetic_user_interruption_", "")
            elif entry_id.startswith("user_interruption_"):
                sample_id = entry_id.replace("user_interruption_", "")
            elif entry_id.isdigit():
                sample_id = entry_id
            if sample_id is None:
                continue
            dest = fdb_prepared / entry_id / "interrupt.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            copied = False
            for ver in vers:
                for folder in (folder_v1_5, folder_v1_0):
                    src_dir = fdb_data_path / ver / folder / sample_id
                    src_json = src_dir / "interrupt.json"
                    if src_json.exists():
                        shutil.copy2(src_json, dest)
                        copied = True
                        break
                    # v1.5 has metadata.json only: {context_text, current_turn_text, timestamps} -> build interrupt.json
                    meta_src = src_dir / "metadata.json"
                    if meta_src.exists():
                        with open(meta_src, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        interrupt_payload = [
                            {
                                "context": meta.get("context_text", ""),
                                "interrupt": meta.get("current_turn_text", ""),
                                "timestamp": meta.get("timestamps", [0.0, 0.0]),
                            }
                        ]
                        with open(dest, "w", encoding="utf-8") as f:
                            json.dump(interrupt_payload, f, indent=2)
                        copied = True
                        break
                if copied:
                    break
            if not copied:
                print(f"Warning: interrupt.json/metadata.json not found for {entry_id} (tried {vers}, {folder_v1_5}/{folder_v1_0})")

    # Behavior eval (background_speech, talking_to_other) needs: input.json, clean_input.json, output.json, clean_output.json
    # Copy input.wav and clean_input.wav from FDB v1.5 data; write placeholder clean_output.json; ASR will fill input.json and clean_input.json
    if subtest in ("background_speech", "talking_to_other") and fdb_data_path and fdb_data_path.exists():
        for entry_id, _ in entries_with_audio:
            entry_id = str(entry_id)
            sample_id = None
            if entry_id.startswith("background_speech_"):
                sample_id = entry_id.replace("background_speech_", "", 1)
            elif entry_id.startswith("talking_to_other_"):
                sample_id = entry_id.replace("talking_to_other_", "", 1)
            if sample_id is None:
                continue
            dest_dir = fdb_prepared / entry_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            for ver in vers:
                src_dir = fdb_data_path / ver / subtest / sample_id
                if not src_dir.exists():
                    continue
                for wav_name in ("input.wav", "clean_input.wav"):
                    src = src_dir / wav_name
                    if src.exists():
                        shutil.copy2(src, dest_dir / wav_name)
                # Placeholder: we don't run model on clean input, so no real clean_output; eval_behavior needs the file to exist
                clean_out = dest_dir / "clean_output.json"
                if not clean_out.exists():
                    with open(clean_out, "w") as f:
                        json.dump({"text": "", "chunks": []}, f, indent=2)
                break
            else:
                print(f"Warning: FDB data not found for {entry_id} (tried {vers}/{subtest}/{sample_id})")

    if run_asr and asr_task and (fdb_repo / "get_transcript" / "asr.py").exists():
        asr_script = fdb_repo / "get_transcript" / "asr.py"
        cmd = [sys.executable, str(asr_script), "--root_dir", str(fdb_prepared), "--task", asr_task]
        if stereo:
            cmd.append("--stereo")
        subprocess.run(cmd, cwd=str(fdb_repo), check=False)
        # For behavior subtests, also produce input.json and clean_input.json from copied input.wav / clean_input.wav
        if subtest in ("background_speech", "talking_to_other"):
            subprocess.run(
                [sys.executable, str(asr_script), "--root_dir", str(fdb_prepared), "--task", "inputs_only"],
                cwd=str(fdb_repo),
                check=False,
            )

    return fdb_prepared


def main():
    parser = argparse.ArgumentParser(description="Prepare FDB-format dir from eval output.jsonl and audio")
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    parser.add_argument("--fdb_repo", type=Path, required=True)
    parser.add_argument("--subdir", default="fdb_prepared")
    parser.add_argument("--run_asr", action="store_true", help="Run FDB ASR (requires --asr_task)")
    parser.add_argument("--asr_task", choices=["full", "user_interruption"], help="FDB asr.py --task")
    parser.add_argument(
        "--fdb_data_path",
        type=Path,
        default=None,
        help="FDB dataset root; required for turn_taking (turn_taking.json) and interruption (interrupt.json)",
    )
    parser.add_argument(
        "--subtest", default=None, help="Subtest name; needed to copy task metadata (turn_taking.json for turn_taking, interrupt.json for interruption)"
    )
    parser.add_argument(
        "--fdb_version", default="v1.0", choices=["v1.0", "v1.5"], help="FDB dataset version (for metadata paths under fdb_data_path)"
    )
    args = parser.parse_args()

    output_jsonl = args.eval_results_dir / "output.jsonl"
    if not output_jsonl.exists():
        print(f"Error: {output_jsonl} not found")
        sys.exit(1)

    path = prepare_fdb_dir(
        args.eval_results_dir,
        output_jsonl,
        args.fdb_repo,
        args.subdir,
        asr_task=args.asr_task,
        run_asr=args.run_asr,
        fdb_data_path=args.fdb_data_path,
        subtest=args.subtest,
        fdb_version=args.fdb_version,
        stereo=True,
    )
    print(f"Prepared: {path}")


if __name__ == "__main__":
    main()
