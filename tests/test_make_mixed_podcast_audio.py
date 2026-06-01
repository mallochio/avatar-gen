from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tests.conftest import load_script

mix = load_script("make_mixed_podcast_audio")


def test_main_merges_and_normalizes(write_wav, tmp_path: Path):
    p1 = write_wav(tmp_path / "a.wav", duration_sec=0.5, left_tone=True)
    p2 = write_wav(tmp_path / "b.wav", duration_sec=0.75, right_tone=True)
    out = tmp_path / "mixed.wav"
    sys.argv = [
        "make_mixed_podcast_audio",
        "--person1-audio",
        p1.as_posix(),
        "--person2-audio",
        p2.as_posix(),
        "--output",
        out.as_posix(),
    ]
    assert mix.main() == 0
    data, sr = sf.read(out)
    assert sr == mix.SAMPLE_RATE
    assert len(data) == int(0.75 * mix.SAMPLE_RATE)
    assert np.max(np.abs(data)) <= 1.0
