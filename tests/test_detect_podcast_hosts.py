from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import load_script

detect = load_script("detect_podcast_hosts")


def test_expand_bbox_clamps_to_image():
    box = detect.expand_bbox(100, 50, 200, 150, width=300, height=200, padding=0.5)
    assert box[0] >= 0
    assert box[1] >= 0
    assert box[2] <= 200
    assert box[3] <= 300


def test_face_landmarks():
    lm = detect.face_landmarks(np.zeros((10, 10, 3), dtype=np.uint8), (10, 20, 30, 40))
    assert lm["face_center_x"] == 20.0
    assert lm["face_width"] == 20.0


def test_detect_faces_fallback_split():
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detect.detect_faces_bgr(bgr)
    assert len(faces) == 2
    assert faces[0][0] < faces[1][0]


def test_detect_hands_heuristic_without_mediapipe(monkeypatch):
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[60:, :] = (20, 80, 200)
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    with pytest.raises((ImportError, AttributeError)):
        __import__("mediapipe")
    # Force import error path by patching import inside detect_hands
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mediapipe":
            raise ImportError("no mediapipe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert isinstance(detect.detect_hands(bgr), bool)


def test_main_writes_spatial_json(two_host_image: Path, tmp_path: Path):
    out = tmp_path / "spatial.json"
    sys.argv = [
        "detect_podcast_hosts",
        "--image",
        two_host_image.as_posix(),
        "--output-json",
        out.as_posix(),
        "--padding",
        "0.2",
    ]
    assert detect.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "bbox" in payload
    assert "person1" in payload["bbox"]
    assert "person2" in payload["bbox"]
    assert payload["image_size"]["width"] == 640


def test_existing_image_rejects_missing(tmp_path: Path):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        detect.existing_image((tmp_path / "missing.png").as_posix())
