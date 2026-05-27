#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def existing_file(path_str: str, exts: set[str]) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    if path.suffix.lower() not in exts:
        allowed = ", ".join(sorted(exts))
        raise argparse.ArgumentTypeError(
            f"Unsupported file type for {path.name}; expected one of: {allowed}"
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", required=True, type=lambda value: existing_file(value, IMAGE_EXTS)
    )
    parser.add_argument(
        "--audio", required=True, type=lambda value: existing_file(value, AUDIO_EXTS)
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt.strip(),
        "cond_image": args.image.as_posix(),
        "cond_audio": {"person1": args.audio.as_posix()},
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
