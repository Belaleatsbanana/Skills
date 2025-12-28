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

"""Prepare NV TTS evaluation datasets.

Reads a config JSON file (local or remote) containing subtest definitions,
fetches manifest JSONL files, and generates nemo-skills test.jsonl files
with manifest content embedded as JSON in user message content.

Usage:
    python prepare.py --config login-eos.nvidia.com:/path/to/evalset_config.json
    python prepare.py --config /local/path/to/evalset_config.json
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

SYSTEM_MESSAGE = "You are a helpful assistant."

# Template for subtest __init__.py files
INIT_TEMPLATE = """# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# NV TTS subtest: {subtest_name}

GENERATION_ARGS = "++prompt_format=openai"
"""


def is_remote_path(path: str) -> bool:
    """Check if path is a remote path (host:/path format)."""
    return ":" in path and not path.startswith("/") and not path.startswith(".")


def fetch_remote_file(remote_path: str, local_path: str) -> None:
    """Fetch a file from a remote host using scp."""
    result = subprocess.run(
        ["scp", remote_path, local_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {remote_path}: {result.stderr}")


def read_file_content(path: str) -> str:
    """Read file content, handling both local and remote paths."""
    if is_remote_path(path):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tmp") as tmp:
            tmp_path = tmp.name
        try:
            fetch_remote_file(path, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def get_remote_host(path: str) -> str:
    """Extract host from a remote path."""
    if is_remote_path(path):
        return path.split(":")[0]
    return ""


def make_remote_path(host: str, path: str) -> str:
    """Create a remote path from host and path."""
    if host:
        return f"{host}:{path}"
    return path


def format_manifest_entry(entry: dict, audio_dir: str) -> dict:
    """Format a manifest entry into nemo-skills format.

    Args:
        entry: Manifest entry with fields like text, context_audio_filepath, etc.
        audio_dir: Base directory for audio files to make paths absolute.

    Returns:
        Formatted entry with messages containing the manifest as JSON string.
    """
    # Make audio paths absolute by combining with audio_dir
    entry_with_absolute_paths = entry.copy()

    if "context_audio_filepath" in entry_with_absolute_paths and audio_dir:
        entry_with_absolute_paths["context_audio_filepath"] = os.path.join(
            audio_dir, entry_with_absolute_paths["context_audio_filepath"]
        )

    if "audio_filepath" in entry_with_absolute_paths and audio_dir:
        entry_with_absolute_paths["audio_filepath"] = os.path.join(
            audio_dir, entry_with_absolute_paths["audio_filepath"]
        )

    # Create the nemo-skills format entry
    content = json.dumps(entry_with_absolute_paths)

    return {
        "problem": "",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ],
    }


def create_subtest_init(subtest_dir: Path, subtest_name: str) -> None:
    """Create __init__.py for a subtest directory."""
    content = INIT_TEMPLATE.format(subtest_name=subtest_name)
    with open(subtest_dir / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def process_subtest(
    subtest_name: str,
    config: dict,
    output_dir: Path,
    remote_host: str,
) -> int:
    """Process a single subtest and generate test.jsonl.

    Args:
        subtest_name: Name of the subtest (e.g., "libritts_seen").
        config: Subtest config with manifest_path, audio_dir, feature_dir.
        output_dir: Base output directory for the dataset.
        remote_host: Remote host for fetching files (empty for local).

    Returns:
        Number of entries processed.
    """
    subtest_dir = output_dir / subtest_name
    subtest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = config["manifest_path"]
    audio_dir = config.get("audio_dir", "")

    # Fetch manifest file
    if remote_host:
        manifest_remote = make_remote_path(remote_host, manifest_path)
        print(f"Fetching manifest from {manifest_remote}...")
        manifest_content = read_file_content(manifest_remote)
    else:
        print(f"Reading manifest from {manifest_path}...")
        manifest_content = read_file_content(manifest_path)

    # Process manifest entries
    output_file = subtest_dir / "test.jsonl"
    count = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for line in manifest_content.strip().split("\n"):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON line: {e}")
                continue

            formatted = format_manifest_entry(entry, audio_dir)
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    # Create __init__.py
    create_subtest_init(subtest_dir, subtest_name)

    print(f"  Wrote {count} entries to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare NV TTS evaluation datasets")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config JSON file (local or remote: host:/path/to/config.json)",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    # Determine if config is remote and extract host
    config_path = args.config
    remote_host = get_remote_host(config_path)

    # Read config file
    print(f"Reading config from {config_path}...")
    config_content = read_file_content(config_path)
    config = json.loads(config_content)

    print(f"Found {len(config)} subtests: {list(config.keys())}")

    total_entries = 0
    for subtest_name, subtest_config in config.items():
        print(f"\nProcessing {subtest_name}...")
        count = process_subtest(subtest_name, subtest_config, output_dir, remote_host)
        total_entries += count

    print(f"\nDone! Processed {total_entries} total entries across {len(config)} subtests.")


if __name__ == "__main__":
    main()
