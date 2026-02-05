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

"""Convert nemo-skills output format to Full-Duplex-Bench format for official scoring."""

import argparse
import json
import re
from pathlib import Path


def clean_special_tokens(text: str) -> str:
    """Remove special timing/frame tokens from S2S model output.

    The S2S model outputs special tokens like:
    - <$X.XX$> - energy/confidence markers
    - <|X.XX|> - timing/duration markers

    These should be stripped for clean text output used in evaluation.
    """
    if not text:
        return text
    # Remove <$X.XX$> patterns (energy/confidence)
    text = re.sub(r'<\$[\d.]+\$>', '', text)
    # Remove <|X.XX|> patterns (timing)
    text = re.sub(r'<\|[\d.]+\|>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def convert_entry(entry: dict, entry_index: int = 0) -> dict:
    """Convert a single nemo-skills entry to Full-Duplex-Bench format.

    nemo-skills format:
        - problem: the prompt/question
        - generation: model's response
        - expected_answer: reference answer (if available)
        - Additional fields: id, category, duration, overlap_duration, etc.

    Full-Duplex-Bench format:
        - prompt: the instruction/question
        - response: model's output
        - reference: expected answer (if available)
        - Additional metadata preserved as-is
    """
    prompt_text = entry.get("problem", "")
    # Clean special tokens from generation (S2S model outputs timing markers)
    generation = clean_special_tokens(entry.get("generation", ""))
    
    converted = {
        "prompt": prompt_text,
        "response": generation,
        "reference": entry.get("expected_answer", ""),
    }

    # Preserve additional metadata fields
    for field in ["id", "category", "duration", "overlap_duration", "audio_path"]:
        if field in entry:
            converted[field] = entry[field]

    return converted


def convert_file(input_path: str, output_path: str) -> int:
    """Convert a nemo-skills JSONL file to Full-Duplex-Bench format.

    Returns the number of entries converted.
    """
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                entry = json.loads(line)
                converted = convert_entry(entry, entry_index=idx)
                entries.append(converted)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Convert nemo-skills output to Full-Duplex-Bench format")
    parser.add_argument("--input", "-i", required=True, help="Path to input nemo-skills JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path to output Full-Duplex-Bench JSONL file")
    parser.add_argument("--subtest", help="Subtest name (for logging)")
    args = parser.parse_args()

    count = convert_file(args.input, args.output)
    print(f"Converted {count} entries from {args.input} to {args.output}")

    if args.subtest:
        print(f"Subtest: {args.subtest}")


if __name__ == "__main__":
    main()
