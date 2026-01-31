#!/usr/bin/env python3

"""Dependency checker for EmergentTTS-Eval integration.

This script is meant to fail fast with a clear actionable message when you are
missing Python packages needed for:
- dataset preparation (`prepare.py`)
- scoring (EmergentTTS-Eval-public `inference.py`)
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


def _try_import(module: str) -> str | None:
    try:
        importlib.import_module(module)
        return None
    except Exception as e:
        return f"{module} ({type(e).__name__}: {e})"


def _venv_install_hint(*, emergent_repo_path: str | None) -> str:
    repo_root = Path(__file__).resolve().parents[4]  # .../nemo_skills/dataset/emergent_tts/scripts
    lines = [
        "To install missing deps into the repo venv:",
        f"  cd {repo_root}",
        "  . ./.venv/bin/activate",
        "  pip install -e .",
        "  pip install librosa soundfile",
    ]
    if emergent_repo_path:
        lines.append(f"  pip install -r {Path(emergent_repo_path).resolve()}/requirements.txt")
    else:
        lines.append("  pip install -r /path/to/EmergentTTS-Eval-public/requirements.txt")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Check dependencies for EmergentTTS-Eval integration")
    p.add_argument("--stage", choices=["prepare", "scoring", "all"], default="all")
    p.add_argument(
        "--emergent_repo_path",
        default=os.environ.get("EMERGENT_TTS_EVAL_REPO", ""),
        help="Path to EmergentTTS-Eval-public (used only to print install hint)",
    )
    args = p.parse_args()

    emergent_repo_path = args.emergent_repo_path or None

    missing: list[str] = []

    if args.stage in ("prepare", "all"):
        for mod in ["datasets", "numpy", "pydub", "tqdm", "librosa", "soundfile"]:
            err = _try_import(mod)
            if err:
                missing.append(err)

    if args.stage in ("scoring", "all"):
        # Minimal set required by EmergentTTS-Eval-public scoring path (fetch-audios mode)
        for mod in [
            "torch",
            "transformers",
            "editdistance",
            "whisper_normalizer",
            "json_repair",
            "tenacity",
            "openai",
            "google.genai",
            "pydub",
            "librosa",
            "soundfile",
        ]:
            err = _try_import(mod)
            if err:
                missing.append(err)

    if missing:
        print("Missing required dependencies:\n")
        for m in missing:
            print(f"- {m}")
        print()
        print(_venv_install_hint(emergent_repo_path=emergent_repo_path))
        raise SystemExit(2)

    print("All required dependencies are available.")


if __name__ == "__main__":
    main()

