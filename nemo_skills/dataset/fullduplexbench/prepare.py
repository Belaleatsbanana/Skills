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

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# Subtest configurations for Full-Duplex-Bench
# Based on the four main evaluation dimensions
SUBTESTS = {
    "pause": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's ability to handle pauses in conversation",
    },
    "backchannel": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's backchanneling behavior (e.g., 'uh-huh', 'yeah')",
    },
    "turn_taking": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's turn-taking capabilities",
    },
    "interruption": {
        "has_reference": True,
        "metrics_type": "exact_match",
        "eval_args": "++eval_type=exact_match",
        "description": "Evaluate model's handling of interruptions",
    },
}

# Template for subtest __init__.py files
INIT_TEMPLATE = """# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

METRICS_TYPE = "{metrics_type}"
GENERATION_ARGS = "++prompt_format=openai"
{eval_args}
"""


def save_audio(audio_data, audio_path, sampling_rate=16000):
    """Save audio data to a WAV file."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio_data, sampling_rate)


def format_entry(entry, subtest_name, config, audio_dir, entry_idx, no_audio=False):
    """Format a single entry for nemo-skills with OpenAI messages format.

    Creates three message variants in a single entry:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)
    """
    prompt_text = entry.get("prompt", entry.get("question", ""))

    formatted = {
        "problem": prompt_text,
    }

    # Add expected answer if available
    if config.get("has_reference") and "reference" in entry:
        formatted["expected_answer"] = entry["reference"]
    elif config.get("has_reference") and "answer" in entry:
        formatted["expected_answer"] = entry["answer"]

    # Preserve additional metadata fields
    for field in ["id", "dataset", "sample_id", "category", "duration", "overlap_duration"]:
        if field in entry:
            formatted[field] = entry[field]

    # System message (shared across all variants)
    system_message = {"role": "system", "content": "You are a helpful assistant."}

    # Handle audio - copy/link files and get audio info
    audio_info = None
    if not no_audio:
        # Check if audio_path is provided (file already exists)
        if "audio_path" in entry:
            import shutil
            from pathlib import Path

            source_audio = Path(entry["audio_path"])
            if source_audio.exists():
                audio_id = f"{subtest_name}_{entry_idx}"
                audio_dest = audio_dir / f"{audio_id}.wav"

                # Copy audio file to our data directory
                shutil.copy(source_audio, audio_dest)

                audio_info = {"audio": {"path": f"fullduplexbench/data/{audio_id}.wav"}}
                formatted["audio_path"] = f"data/{audio_id}.wav"

        # Handle direct audio data (for compatibility)
        elif "audio" in entry and entry["audio"] is not None:
            audio_id = f"{subtest_name}_{entry_idx}"
            audio_path = audio_dir / f"{audio_id}.wav"

            # Handle different audio data formats
            if isinstance(entry["audio"], dict) and "array" in entry["audio"]:
                save_audio(entry["audio"]["array"], audio_path, entry["audio"].get("sampling_rate", 16000))
            else:
                # If audio is already a numpy array or similar
                save_audio(entry["audio"], audio_path)

            audio_info = {"audio": {"path": f"fullduplexbench/data/{audio_id}.wav"}}
            formatted["audio_path"] = f"data/{audio_id}.wav"

    # Create three message variants:

    # 1. messages: audio only (empty content, with audio)
    user_message_audio = {"role": "user", "content": ""}
    if audio_info:
        user_message_audio.update(audio_info)
    formatted["messages"] = [system_message.copy(), user_message_audio]

    # 2. messages_text_audio: both text and audio
    user_message_text_audio = {"role": "user", "content": prompt_text}
    if audio_info:
        user_message_text_audio.update(audio_info)
    formatted["messages_text_audio"] = [system_message.copy(), user_message_text_audio]

    # 3. messages_text: text only (no audio)
    user_message_text = {"role": "user", "content": prompt_text}
    formatted["messages_text"] = [system_message.copy(), user_message_text]

    return formatted


def create_subtest_init(subtest_dir, config):
    """Create __init__.py for a subtest directory."""
    eval_args_line = f'EVAL_ARGS = "{config["eval_args"]}"' if config["eval_args"] else ""
    content = INIT_TEMPLATE.format(
        metrics_type=config["metrics_type"],
        eval_args=eval_args_line,
    )
    with open(subtest_dir / "__init__.py", "w") as f:
        f.write(content)


def process_subtest(subtest_name, config, data_dir, audio_dir, fdb_data_path, no_audio=False):
    """Process a single subtest and save to JSONL.

    Each entry contains three message variants:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)

    Full-Duplex-Bench v1.0 structure:
    - candor_pause_handling/{ID}/input.wav, pause.json, transcription.json
    - candor_turn_taking/{ID}/input.wav, turn_taking.json, transcription.json
    - icc_backchannel/{ID}/input.wav, transcription.json
    - synthetic_pause_handling/{ID}/input.wav, pause.json, transcription.json
    - synthetic_user_interruption/{ID}/input.wav, context.wav, interrupt.wav, interrupt.json
    """
    subtest_dir = data_dir / subtest_name
    subtest_dir.mkdir(parents=True, exist_ok=True)

    output_file = subtest_dir / "test.jsonl"
    entries = []
    entry_idx = 0

    print(f"Processing {subtest_name}...")
    print(f"  Description: {config['description']}")

    # Map subtests to Full-Duplex-Bench dataset folders
    subtest_mapping = {
        "pause": ["candor_pause_handling", "synthetic_pause_handling"],
        "backchannel": ["icc_backchannel"],
        "turn_taking": ["candor_turn_taking"],
        "interruption": ["synthetic_user_interruption"],
    }

    dataset_folders = subtest_mapping.get(subtest_name, [])
    if not dataset_folders:
        print(f"  Warning: Unknown subtest {subtest_name}")
        return 0

    # Load test cases from each dataset folder
    test_cases = []
    for folder_name in dataset_folders:
        # Try different possible paths
        possible_paths = [
            fdb_data_path / "v1.0" / folder_name,  # Standard: v1.0
            fdb_data_path / "v1_0" / folder_name,  # Alternative: v1_0
            fdb_data_path / folder_name,  # Direct
        ]

        folder_path = None
        for path in possible_paths:
            if path.exists():
                folder_path = path
                break

        if folder_path is None:
            print("  Warning: Dataset folder not found. Tried:")
            for path in possible_paths:
                print(f"    - {path}")
            continue

        # Find all sample directories (numeric IDs)
        sample_dirs = sorted([d for d in folder_path.iterdir() if d.is_dir()])
        print(f"  Found {len(sample_dirs)} samples in {folder_name}")

        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            input_wav = sample_dir / "input.wav"

            if not input_wav.exists():
                print(f"  Warning: input.wav not found in {sample_dir}")
                continue

            # Load transcription if available
            transcription_file = sample_dir / "transcription.json"
            transcription = ""
            if transcription_file.exists():
                with open(transcription_file, "r") as f:
                    trans_data = json.load(f)
                    # Extract text from transcription
                    if isinstance(trans_data, dict):
                        transcription = trans_data.get("text", "")
                    elif isinstance(trans_data, list) and len(trans_data) > 0:
                        # Word-level transcription
                        transcription = " ".join([w.get("word", "") for w in trans_data])

            # Create test case
            test_case = {
                "id": f"{folder_name}_{sample_id}",
                "audio_path": str(input_wav),
                "prompt": transcription or "Respond to the user's speech in the audio.",
                "dataset": folder_name,
                "sample_id": sample_id,
            }

            test_cases.append(test_case)

    if not test_cases:
        print("  No test cases found. Make sure you downloaded the dataset from:")
        print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
        return 0

    # Process each test case
    for entry in tqdm(test_cases, desc="  Processing entries"):
        formatted = format_entry(
            entry,
            subtest_name,
            config,
            audio_dir,
            entry_idx,
            no_audio=no_audio,
        )
        entries.append(formatted)
        entry_idx += 1

    # Write JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Create __init__.py
    create_subtest_init(subtest_dir, config)

    print(f"  Wrote {len(entries)} entries to {output_file}")
    return len(entries)


# All five v1.0 zip names on Google Drive (must all be downloaded and extracted for full dataset)
EXPECTED_V1_ZIPS = [
    "candor_pause_handling.zip",
    "candor_turn_taking.zip",
    "icc_backchannel.zip",
    "synthetic_pause_handling.zip",
    "synthetic_user_interruption.zip",
]


def download_dataset(download_dir: Path) -> bool:
    """Download only the Full-Duplex-Bench v1.0 subfolder from Google Drive (not v1.5),
    then unzip all archives so that all five dataset folders are present.

    Original dataset with v1.0 and v1.5: https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3
    We download only the v1.0 folder into download_dir/v1.0, then extract every .zip found there.

    Returns:
        True if download successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("\nError: 'gdown' package not found. Install it with:")
        print("  pip install gdown")
        return False

    import shutil
    import zipfile

    # v1.0 subfolder only (do not download v1.5)
    v1_0_folder_url = "https://drive.google.com/drive/folders/1hxzRk7xgtdr5ZEoctnp0sFK0COv91W3h"
    v1_dir = download_dir / "v1.0"

    print("\n" + "=" * 60)
    print("Downloading Full-Duplex-Bench v1.0 only from Google Drive")
    print("=" * 60)
    print(f"Source: {v1_0_folder_url}")
    print(f"Destination: {v1_dir}")
    print(
        "All 5 zips will be downloaded and extracted (candor_pause_handling, candor_turn_taking, icc_backchannel, synthetic_pause_handling, synthetic_user_interruption)."
    )
    print("=" * 60 + "\n")

    try:
        v1_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading v1.0 dataset (all zip files in folder)...")
        gdown.download_folder(
            url=v1_0_folder_url,
            output=str(v1_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )

        # gdown may create a single subfolder with the folder name; collect zips from v1_dir and one level down
        zip_files = list(v1_dir.glob("*.zip")) or list(v1_dir.rglob("*.zip"))
        if not zip_files and any(v1_dir.iterdir()):
            sub = next((d for d in v1_dir.iterdir() if d.is_dir()), None)
            if sub:
                sub_zips = list(sub.rglob("*.zip"))
                if sub_zips:
                    for z in sub_zips:
                        dest = v1_dir / z.name
                        if not dest.exists() or dest.stat().st_size != z.stat().st_size:
                            shutil.move(str(z), str(dest))
                    zip_files = list(v1_dir.glob("*.zip"))

        print("\n" + "=" * 60)
        print("Download completed.")
        print("=" * 60 + "\n")

        if not zip_files:
            zip_files = list(v1_dir.glob("*.zip")) or list(v1_dir.rglob("*.zip"))
        if not zip_files:
            print("Warning: No .zip files found under v1.0. Check Drive permissions or download manually.")
            return True

        # Extract all zip files in v1.0
        print(f"Extracting {len(zip_files)} ZIP file(s) in v1.0...")
        for zip_file in tqdm(sorted(zip_files), desc="Extracting"):
            try:
                extract_dir = zip_file.parent
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                zip_file.unlink()
                print(f"  Extracted: {zip_file.name}")
            except Exception as e:
                print(f"  Warning: Failed to extract {zip_file.name}: {e}")

        print(f"\nExtracted {len(zip_files)} ZIP file(s).")

        # Verify expected folders exist (names without .zip)
        expected_folders = [p.replace(".zip", "") for p in EXPECTED_V1_ZIPS]
        found = [d.name for d in v1_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        missing = [f for f in expected_folders if f not in found]
        if missing:
            print(f"Warning: After extraction, missing folder(s): {missing}")
            print("  Re-run prepare (without --fdb_data_path to re-download) or add the missing zip(s) manually to:")
            print(f"  {v1_dir}")
        else:
            print("All expected v1.0 dataset folders are present.")

        return True

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nYou can manually download from:")
        print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
        print("  Open the v1.0 subfolder, download all 5 zips, and extract them into:")
        print(f"  {download_dir / 'v1.0'}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Full-Duplex-Bench dataset for nemo-skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download v1.0 to default path (s2s/Full-Duplex-Bench-data) and prepare
  python prepare.py

  # Use existing dataset (no download)
  python prepare.py --fdb_data_path /path/to/Full-Duplex-Bench-data

  # Download to a specific location and prepare
  python prepare.py --fdb_data_path /path/to/download/location
        """,
    )
    parser.add_argument(
        "--fdb_data_path",
        type=str,
        default=None,
        help="Path to Full-Duplex-Bench dataset. If not set, downloads v1.0 to s2s/Full-Duplex-Bench-data and prepares.",
    )
    parser.add_argument(
        "--subtests",
        nargs="+",
        default=None,
        help="Specific subtests to process (default: all)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip processing audio files (faster, for testing)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent
    audio_dir = data_dir / "data"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _skills_dir = data_dir.parent.parent.parent
    _s2s_root = _skills_dir.parent
    _default_fdb_data = _s2s_root / "Full-Duplex-Bench-data"

    if args.fdb_data_path:
        fdb_data_path = Path(args.fdb_data_path)
        if not fdb_data_path.exists():
            print(f"\nError: Full-Duplex-Bench data path not found: {fdb_data_path}")
            print("Run without --fdb_data_path to download, or use:")
            print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
            return
    else:
        download_path = _default_fdb_data
        print(f"Downloading v1.0 to {download_path} (overwriting if present)...")
        if not download_dataset(download_path):
            print("\nDownload failed. Exiting.")
            return
        fdb_data_path = download_path

    subtests_to_process = args.subtests if args.subtests else list(SUBTESTS.keys())

    print(f"Processing {len(subtests_to_process)} subtests...")
    if args.no_audio:
        print("Skipping audio download (--no-audio)")

    total_entries = 0
    for subtest_name in subtests_to_process:
        if subtest_name not in SUBTESTS:
            print(f"Warning: Unknown subtest '{subtest_name}', skipping")
            continue

        config = SUBTESTS[subtest_name]
        count = process_subtest(subtest_name, config, data_dir, audio_dir, fdb_data_path, no_audio=args.no_audio)
        total_entries += count

    print(f"\nDone! Processed {total_entries} total entries across {len(subtests_to_process)} subtests.")


if __name__ == "__main__":
    main()
