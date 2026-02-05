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
import sys
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
            from pathlib import Path
            import shutil
            
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
            fdb_data_path / folder_name,            # Direct
        ]
        
        folder_path = None
        for path in possible_paths:
            if path.exists():
                folder_path = path
                break
        
        if folder_path is None:
            print(f"  Warning: Dataset folder not found. Tried:")
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
                "prompt": transcription or f"Respond to the user's speech in the audio.",
                "dataset": folder_name,
                "sample_id": sample_id,
            }
            
            test_cases.append(test_case)
    
    if not test_cases:
        print(f"  No test cases found. Make sure you downloaded the dataset from:")
        print(f"  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
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


def download_dataset(download_dir: Path) -> bool:
    """Download Full-Duplex-Bench v1.0 dataset from Google Drive.
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("\nError: 'gdown' package not found. Install it with:")
        print("  pip install gdown")
        return False
    
    import zipfile
    
    # Google Drive folder ID for Full-Duplex-Bench v1.0 dataset
    # This is the v1.0 subfolder, not the root folder
    folder_url = "https://drive.google.com/drive/folders/1hxzRk7xgtdr5ZEoctnp0sFK0COv91W3h"
    
    print("\n" + "=" * 60)
    print("Downloading Full-Duplex-Bench v1.0 from Google Drive")
    print("=" * 60)
    print(f"Source: {folder_url}")
    print(f"Destination: {download_dir}/v1.0")
    print("\nDownloading ~500MB (v1.0 only)...")
    print("=" * 60 + "\n")
    
    try:
        # Create download directory
        v1_dir = download_dir / "v1.0"
        v1_dir.mkdir(parents=True, exist_ok=True)
        
        # Download v1.0 folder from Google Drive
        print("Downloading v1.0 dataset files...")
        gdown.download_folder(
            url=folder_url,
            output=str(v1_dir),
            quiet=False,
            use_cookies=False,
        )
        
        print("\n" + "=" * 60)
        print("Download completed successfully!")
        print("=" * 60 + "\n")
        
        # Extract all zip files in v1.0
        print("Extracting ZIP files...")
        zip_files = list(v1_dir.rglob("*.zip"))
        if zip_files:
            for zip_file in tqdm(zip_files, desc="Extracting"):
                try:
                    # Extract to the same directory as the zip file
                    extract_dir = zip_file.parent
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Remove the zip file after extraction
                    zip_file.unlink()
                    print(f"  Extracted: {zip_file.name}")
                except Exception as e:
                    print(f"  Warning: Failed to extract {zip_file.name}: {e}")
            
            print(f"\nExtracted {len(zip_files)} ZIP files successfully!")
        
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nYou can manually download from:")
        print(f"  {folder_url}")
        print(f"And extract to: {download_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Full-Duplex-Bench dataset for nemo-skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-download and prepare dataset
  python prepare.py --download
  
  # Use existing downloaded dataset
  python prepare.py --fdb_data_path /path/to/dataset
  
  # Download to specific location and prepare
  python prepare.py --download --fdb_data_path /path/to/download/location
        """
    )
    parser.add_argument(
        "--fdb_data_path",
        type=str,
        default=None,
        help="Path to Full-Duplex-Bench dataset directory. If --download is used, this is where data will be downloaded.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Automatically download dataset from Google Drive (requires 'gdown' package: pip install gdown)",
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
    
    # Determine dataset path
    if args.download:
        # Download dataset
        if args.fdb_data_path:
            download_path = Path(args.fdb_data_path)
        else:
            # Default download location
            download_path = data_dir.parent.parent.parent / "Full-Duplex-Bench-data"
        
        print(f"Will download dataset to: {download_path}")
        
        # Check if already exists
        if download_path.exists() and any(download_path.iterdir()):
            response = input(f"\nDirectory {download_path} already exists. Re-download? [y/N]: ")
            if response.lower() != 'y':
                print("Using existing dataset...")
            else:
                if not download_dataset(download_path):
                    print("\nDownload failed. Exiting.")
                    return
        else:
            if not download_dataset(download_path):
                print("\nDownload failed. Exiting.")
                return
        
        fdb_data_path = download_path
    else:
        # Use provided path
        if not args.fdb_data_path:
            print("\nError: Either --download or --fdb_data_path must be provided.")
            print("\nOptions:")
            print("  1. Auto-download: python prepare.py --download")
            print("  2. Use existing:  python prepare.py --fdb_data_path /path/to/dataset")
            print("\nFor manual download, get the dataset from:")
            print("  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
            return
        
        fdb_data_path = Path(args.fdb_data_path)

    if not fdb_data_path.exists():
        print(f"\nError: Full-Duplex-Bench data path not found: {fdb_data_path}")
        print("\nOptions:")
        print("  1. Auto-download: python prepare.py --download")
        print("  2. Manual download from: https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3")
        return

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
