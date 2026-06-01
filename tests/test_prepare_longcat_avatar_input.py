from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.conftest import load_script

prepare = load_script("prepare_longcat_avatar_input")


def test_existing_file_validates_extensions(tmp_path: Path, write_wav):
    wav = write_wav(tmp_path / "ok.wav")
    assert prepare.existing_file(wav.as_posix(), prepare.AUDIO_EXTS) == wav.resolve()
    with pytest.raises(__import__("argparse").ArgumentTypeError):
        prepare.existing_file((tmp_path / "bad.txt").as_posix(), prepare.AUDIO_EXTS)


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
