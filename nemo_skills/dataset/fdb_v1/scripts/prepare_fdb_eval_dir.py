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
from typing import Optional

try:
    import soundfile as sf
except ImportError:
    sf = None


def _version_folders(fdb_version: str):
    """Return (primary, alternate) folder names for the given FDB version (e.g. v1.0 -> ('v1.0', 'v1_0'))."""
    if fdb_version == "v1.5":
        return ("v1.5", "v1_5")
    return ("v1.0", "v1_0")


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
) -> Path:
    """Build FDB-format dir under eval_results_dir/subdir_name. Returns path to prepared dir."""
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
            if e.isdigit():
                return e
        return str(entry_id)

    def ensure_mono_wav(src: Path, dest: Path) -> None:
        """Copy wav to dest; if multi-channel, convert to mono (FDB backchannel Silero VAD requires mono)."""
        if sf is None:
            shutil.copy2(src, dest)
            return
        try:
            data, sr = sf.read(str(src))
            if data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(str(dest), data, sr)
        except Exception:
            shutil.copy2(src, dest)

    fdb_prepared.mkdir(parents=True, exist_ok=True)
    for entry_id, src_wav in entries_with_audio:
        dest_dir = fdb_prepared / fdb_dir_name(entry_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_wav = dest_dir / "output.wav"
        ensure_mono_wav(src_wav, dest_wav)

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
    if subtest == "interruption" and fdb_data_path and fdb_data_path.exists():
        for entry_id, _ in entries_with_audio:
            entry_id = str(entry_id)
            if entry_id.startswith("synthetic_user_interruption_"):
                sample_id = entry_id.replace("synthetic_user_interruption_", "")
            elif entry_id.isdigit():
                sample_id = entry_id
            else:
                continue
            for ver in vers:
                src = fdb_data_path / ver / "synthetic_user_interruption" / sample_id / "interrupt.json"
                if src.exists():
                    shutil.copy2(src, fdb_prepared / entry_id / "interrupt.json")
                    break
            else:
                print(f"Warning: interrupt.json not found for {entry_id} (tried {vers})")

    if run_asr and asr_task and (fdb_repo / "get_transcript" / "asr.py").exists():
        asr_script = fdb_repo / "get_transcript" / "asr.py"
        cmd = [sys.executable, str(asr_script), "--root_dir", str(fdb_prepared), "--task", asr_task]
        subprocess.run(cmd, cwd=str(fdb_repo), check=False)

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
    )
    print(f"Prepared: {path}")


if __name__ == "__main__":
    main()
