#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _extract_user_json(record: dict) -> dict | None:
    for msg in record.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
    return None


def _link_or_copy(src: str, dst: str, mode: str):
    if mode == "symlink":
        if os.path.islink(dst):
            if os.readlink(dst) == src:
                return
            os.unlink(dst)
        elif os.path.exists(dst):
            os.unlink(dst)
        os.symlink(src, dst)
        return

    if mode == "copy":
        shutil.copyfile(src, dst)
        return

    raise ValueError(f"Unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser(description="Convert NeMo-Skills TTS outputs into Emergent audio layout")
    p.add_argument("--ns_output_jsonl", required=True, help="Path to NeMo-Skills output.jsonl")
    p.add_argument("--out_dir", required=True, help="Destination directory for <unique_id_eval>.wav")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    missing = 0

    with open(args.ns_output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user_json = _extract_user_json(record) or {}
            unique_id = user_json.get("unique_id_eval", record.get("unique_id_eval"))
            audio_path = (record.get("audio") or {}).get("path")

            if unique_id is None:
                skipped += 1
                continue
            if not audio_path or not os.path.exists(audio_path):
                missing += 1
                continue

            dst = out_dir / f"{unique_id}.wav"
            if dst.exists() and not args.overwrite:
                continue
            _link_or_copy(audio_path, str(dst), args.mode)
            converted += 1

    print(
        f"Converted {converted} files into {out_dir}. "
        f"skipped(no unique_id_eval)={skipped}, missing_audio={missing}"
    )


if __name__ == "__main__":
    main()

