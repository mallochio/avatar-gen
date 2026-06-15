from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from avatar_segments import audio_duration_sec, compute_num_segments


def test_compute_num_segments_short_clip():
    assert compute_num_segments(2.0) == 1


def test_compute_num_segments_long_clip():
    assert compute_num_segments(20.0) >= 5


def test_compute_num_segments_at_first_clip_boundary():
    first_clip = 93 / 25
    assert compute_num_segments(first_clip) == 1
    assert compute_num_segments(first_clip + 0.01) >= 2


def test_audio_duration_sec_wav(write_wav, tmp_path: Path):
    wav = write_wav(tmp_path / "clip.wav", duration_sec=4.5)
    duration = audio_duration_sec(wav)
    assert duration is not None
    assert abs(duration - 4.5) < 0.05


def test_audio_duration_sec_wav_fallback_when_librosa_fails(
    monkeypatch, write_wav, tmp_path: Path
):
    wav = write_wav(tmp_path / "clip.wav", duration_sec=2.5)
    broken = MagicMock()
    broken.load.side_effect = RuntimeError("forced")
    monkeypatch.setitem(sys.modules, "librosa", broken)
    duration = audio_duration_sec(wav)
    assert duration is not None
    assert abs(duration - 2.5) < 0.05


def test_audio_duration_sec_non_wav_returns_none_when_librosa_fails(
    monkeypatch, tmp_path: Path
):
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fake")
    broken = MagicMock()
    broken.load.side_effect = RuntimeError("forced")
    monkeypatch.setitem(sys.modules, "librosa", broken)
    assert audio_duration_sec(clip) is None
