#!/usr/bin/env python3
"""Detect two-host bounding boxes and optional landmarks from a podcast seed image.

Bounding boxes use LongCat multi-person format:
  [y_min, x_min, y_max, x_max]  (y = height, x = width)

Hosts are ordered left-to-right as person1 (Host A) and person2 (Host B).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def existing_image(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise argparse.ArgumentTypeError(f"Unsupported image type: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=existing_image)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.35,
        help="Expand face box fraction for upper-body framing (default: 0.35)",
    )
    return parser.parse_args()


def _load_face_detector() -> cv2.FaceDetectorYN | None:
    repo = Path(cv2.data.haarcascades)
    haar = repo / "haarcascade_frontalface_default.xml"
    if haar.is_file():
        cascade = cv2.CascadeClassifier(haar.as_posix())
        return cascade  # type: ignore[return-value]
    return None


def detect_faces_bgr(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = _load_face_detector()
    faces: list[tuple[int, int, int, int]] = []
    if detector is not None and hasattr(detector, "detectMultiScale"):
        found = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        for x, y, w, h in found:
            faces.append((int(x), int(y), int(x + w), int(y + h)))
    if len(faces) >= 2:
        faces.sort(key=lambda b: b[0])
        return faces[:2]

    # Fallback: split frame for two seated hosts (matches upstream default heuristic)
    h, w = image_bgr.shape[:2]
    face_scale = 0.1
    y_min = int(h * face_scale)
    y_max = int(h * (1 - face_scale))
    half_w = w // 2
    x_pad = int(half_w * face_scale)
    left = (x_pad, y_min, half_w - x_pad, y_max)
    right = (half_w + x_pad, y_min, w - x_pad, y_max)
    return [left, right]


def expand_bbox(
    x0: int, y0: int, x1: int, y1: int, width: int, height: int, padding: float
) -> list[int]:
    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    y_min = max(0, y0 - pad_y)
    x_min = max(0, x0 - pad_x)
    y_max = min(height, y1 + pad_y * 2)
    x_max = min(width, x1 + pad_x)
    return [y_min, x_min, y_max, x_max]


def detect_hands(image_bgr: np.ndarray) -> bool:
    """Return True when visible hands are likely in the seed frame."""
    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        # Heuristic: lower third skin-tone density as weak proxy
        h, w = image_bgr.shape[:2]
        lower = image_bgr[int(h * 0.55) :, :]
        hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
        return float(mask.mean()) > 12.0

    mp_hands = mp.solutions.hands
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.4) as hands:
        result = hands.process(rgb)
        return bool(result.multi_hand_landmarks)


def face_landmarks(image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> dict[str, float]:
    x0, y0, x1, y1 = bbox_xyxy
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return {
        "face_center_x": cx,
        "face_center_y": cy,
        "face_width": float(x1 - x0),
        "face_height": float(y1 - y0),
    }


def main() -> int:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    faces = detect_faces_bgr(bgr)
    person1_box = expand_bbox(*faces[0], width, height, args.padding)
    person2_box = expand_bbox(*faces[1], width, height, args.padding)

    payload = {
        "image": args.image.as_posix(),
        "image_size": {"width": width, "height": height},
        "bbox": {
            "person1": person1_box,
            "person2": person2_box,
        },
        "landmarks": {
            "person1": face_landmarks(bgr, faces[0]),
            "person2": face_landmarks(bgr, faces[1]),
        },
        "has_visible_hands": detect_hands(bgr),
        "bbox_format": "[y_min, x_min, y_max, x_max]",
        "routing_note": (
            "Map person1 bbox to Host A audio; person2 bbox to Host B audio. "
            "Non-speaking host audio must be silent on that track."
        ),
    }

    out = Path(args.output_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
