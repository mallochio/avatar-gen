from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from avatar_segments import compute_num_segments
from tests.conftest import load_script

prepare_multi = load_script("prepare_longcat_avatar_multi_input")


def test_compute_num_segments_short_clip():
    assert compute_num_segments(2.0) == 1


def test_compute_num_segments_long_clip():
    # 93/25 = 3.72s first clip; each extra ~3.2s
    assert compute_num_segments(20.0) >= 5


def test_existing_file_rejects_bad_extension(tmp_path: Path):
    bad = tmp_path / "face.gif"
    bad.write_bytes(b"gif")
    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported"):
        prepare_multi.existing_file(bad.as_posix(), prepare_multi.IMAGE_EXTS)


def test_main_with_spatial_json(two_host_image: Path, write_wav, tmp_path: Path):
    a = write_wav(tmp_path / "p1.wav", duration_sec=8.0)
    b = write_wav(tmp_path / "p2.wav", duration_sec=8.0)
    spatial = tmp_path / "spatial.json"
    spatial.write_text(
        json.dumps(
            {
                "bbox": {"person1": [0, 0, 100, 100], "person2": [0, 100, 100, 200]},
                "has_visible_hands": True,
                "duration_sec": 8.0,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "multi.json"
    sys.argv = [
        "prepare_longcat_avatar_multi_input",
        "--image",
        two_host_image.as_posix(),
        "--person1-audio",
        a.as_posix(),
        "--person2-audio",
        b.as_posix(),
        "--prompt",
        "Podcast hosts talking.",
        "--spatial-json",
        spatial.as_posix(),
        "--audio-type",
        "add",
        "--output-json",
        out.as_posix(),
    ]
    assert prepare_multi.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["audio_type"] == "add"
    assert payload["bbox"]["person1"] == [0, 0, 100, 100]
    assert payload["first_frame_hand_presence_check"] is True
    assert payload["recommended_num_segments"] >= 2


def test_main_loads_duration_from_audio(two_host_image: Path, write_wav, tmp_path: Path):
    a = write_wav(tmp_path / "p1.wav", duration_sec=4.0)
    b = write_wav(tmp_path / "p2.wav", duration_sec=4.0)
    out = tmp_path / "multi2.json"
    sys.argv = [
        "prepare_longcat_avatar_multi_input",
        "--image",
        two_host_image.as_posix(),
        "--person1-audio",
        a.as_posix(),
        "--person2-audio",
        b.as_posix(),
        "--prompt",
        "Test.",
        "--output-json",
        out.as_posix(),
    ]
    assert prepare_multi.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "recommended_num_segments" in payload
