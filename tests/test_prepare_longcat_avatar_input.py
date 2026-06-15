from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from avatar_segments import compute_num_segments
from tests.conftest import load_script

prepare = load_script("prepare_longcat_avatar_input")


def test_existing_file_validates_extensions(tmp_path: Path, write_wav):
    wav = write_wav(tmp_path / "ok.wav")
    assert prepare.existing_file(wav.as_posix(), prepare.AUDIO_EXTS) == wav.resolve()
    with pytest.raises(argparse.ArgumentTypeError):
        prepare.existing_file((tmp_path / "bad.txt").as_posix(), prepare.AUDIO_EXTS)


def test_existing_file_rejects_missing():
    with pytest.raises(argparse.ArgumentTypeError, match="not found"):
        prepare.existing_file("/nonexistent/path.png", prepare.IMAGE_EXTS)


def test_existing_file_rejects_bad_image_extension(tmp_path: Path):
    bad = tmp_path / "face.gif"
    bad.write_bytes(b"gif")
    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported"):
        prepare.existing_file(bad.as_posix(), prepare.IMAGE_EXTS)


def test_main_rejects_invalid_image_extension(
    two_host_image: Path, write_wav, tmp_path: Path
):
    audio = write_wav(tmp_path / "voice.wav")
    bad_image = tmp_path / "bad.gif"
    bad_image.write_bytes(b"gif")
    out = tmp_path / "input.json"
    sys.argv = [
        "prepare_longcat_avatar_input",
        "--image",
        bad_image.as_posix(),
        "--audio",
        audio.as_posix(),
        "--prompt",
        "Test.",
        "--output-json",
        out.as_posix(),
    ]
    with pytest.raises(SystemExit):
        prepare.main()


def test_main_writes_single_host_json(two_host_image: Path, write_wav, tmp_path: Path):
    audio = write_wav(tmp_path / "voice.wav")
    out = tmp_path / "input.json"
    sys.argv = [
        "prepare_longcat_avatar_input",
        "--image",
        two_host_image.as_posix(),
        "--audio",
        audio.as_posix(),
        "--prompt",
        "  Hello world.  ",
        "--output-json",
        out.as_posix(),
    ]
    assert prepare.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["prompt"] == "Hello world."
    assert payload["cond_audio"]["person1"] == audio.resolve().as_posix()
    assert payload["recommended_num_segments"] == 1


def test_main_recommends_segments_for_long_audio(two_host_image: Path, write_wav, tmp_path: Path):
    audio = write_wav(tmp_path / "voice.wav", duration_sec=20.0)
    out = tmp_path / "input.json"
    sys.argv = [
        "prepare_longcat_avatar_input",
        "--image",
        two_host_image.as_posix(),
        "--audio",
        audio.as_posix(),
        "--prompt",
        "Long clip.",
        "--output-json",
        out.as_posix(),
    ]
    assert prepare.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["recommended_num_segments"] == compute_num_segments(20.0)
