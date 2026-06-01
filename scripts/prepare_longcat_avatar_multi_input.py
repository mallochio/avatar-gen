#!/usr/bin/env python3
"""Build LongCat multi-person avatar input JSON for a two-host podcast."""
from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, type=lambda v: existing_file(v, IMAGE_EXTS)
    )
    parser.add_argument(
        "--person1-audio", required=True, type=lambda v: existing_file(v, AUDIO_EXTS)
    )
    parser.add_argument(
        "--person2-audio", required=True, type=lambda v: existing_file(v, AUDIO_EXTS)
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--spatial-json",
        default=None,
        help="JSON from detect_podcast_hosts.py (bbox + has_visible_hands)",
    )
    parser.add_argument(
        "--audio-type",
        default="para",
        choices=["para", "add"],
        help="LongCat multi audio layout (default: para for simultaneous hosts)",
    )
    return parser.parse_args()


def compute_num_segments(duration_sec: float, fps: int = 25, num_frames: int = 93, num_cond_frames: int = 13) -> int:
    first_clip = num_frames / fps
    if duration_sec <= first_clip:
        return 1
    per_segment = (num_frames - num_cond_frames) / fps
    remaining = duration_sec - first_clip
    extra = math.ceil(remaining / per_segment)
    return 1 + extra


def main() -> int:
    args = parse_args()
    spatial: dict = {}
    if args.spatial_json:
        spatial_path = Path(args.spatial_json).expanduser().resolve()
        spatial = json.loads(spatial_path.read_text(encoding="utf-8"))

    payload: dict = {
        "prompt": args.prompt.strip(),
        "cond_image": args.image.as_posix(),
        "cond_audio": {
            "person1": args.person1_audio.as_posix(),
            "person2": args.person2_audio.as_posix(),
        },
        "audio_type": args.audio_type,
    }

    bbox = spatial.get("bbox")
    if bbox:
        payload["bbox"] = bbox

    duration_sec = spatial.get("duration_sec")
    if duration_sec is None:
        try:
            import librosa

            y, sr = librosa.load(args.person1_audio.as_posix(), sr=None, mono=True)
            duration_sec = len(y) / float(sr)
        except Exception:
            duration_sec = None

    if duration_sec is not None:
        payload["recommended_num_segments"] = compute_num_segments(float(duration_sec))

    if spatial.get("has_visible_hands"):
        payload["first_frame_hand_presence_check"] = True

    out = Path(args.output_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    if "recommended_num_segments" in payload:
        print(f"recommended_num_segments={payload['recommended_num_segments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
