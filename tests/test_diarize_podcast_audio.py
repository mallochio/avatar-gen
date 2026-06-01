from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import load_script

diarize = load_script("diarize_podcast_audio")


def test_choose_backend_explicit():
    assert diarize.choose_backend("mfcc") == "mfcc"
    assert diarize.choose_backend("pyannote") == "pyannote"


def test_choose_backend_auto_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert diarize.choose_backend("auto") == "mfcc"


def test_labels_to_tracks_splits_by_speaker():
    n_frames = 4
    y = np.ones(n_frames * diarize.HOP_LENGTH, dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    p1, p2 = diarize.labels_to_tracks(y, labels)
    assert p1[: diarize.HOP_LENGTH].sum() > 0
    assert p2[2 * diarize.HOP_LENGTH : 3 * diarize.HOP_LENGTH].sum() > 0
    assert p2[: diarize.HOP_LENGTH].sum() == 0


def test_diarize_mfcc_short_audio_fallback():
    y = np.zeros(diarize.SAMPLE_RATE, dtype=np.float32)
    y[:8000] = 0.2 * np.sin(np.linspace(0, 40 * np.pi, 8000))
    labels, speech_mask = diarize.diarize_mfcc(y)
    assert len(labels) == len(diarize._mfcc_features(y))
    assert labels.max() <= 1


def test_diarize_mfcc_two_segment_audio(write_wav, tmp_path: Path):
    left = write_wav(tmp_path / "left.wav", duration_sec=0.8, left_tone=True, right_tone=False)
    right = write_wav(tmp_path / "right.wav", duration_sec=0.8, left_tone=False, right_tone=True)
    mix_mod = load_script("make_mixed_podcast_audio")
    mixed = tmp_path / "mix.wav"
    sys.argv = [
        "make_mixed_podcast_audio",
        "--person1-audio",
        left.as_posix(),
        "--person2-audio",
        right.as_posix(),
        "--output",
        mixed.as_posix(),
    ]
    mix_mod.main()

    y, _ = __import__("librosa").load(mixed.as_posix(), sr=diarize.SAMPLE_RATE, mono=True)
    labels, _ = diarize.diarize_mfcc(y)
    p1, p2 = diarize.labels_to_tracks(y, labels)
    assert np.max(np.abs(p1)) > 0
    assert np.max(np.abs(p2)) > 0


def test_main_writes_outputs(write_wav, tmp_path: Path):
    mixed = write_wav(tmp_path / "mixed.wav", duration_sec=1.2, left_tone=True, right_tone=True)
    out_dir = tmp_path / "out"
    meta = tmp_path / "meta.json"
    sys.argv = [
        "diarize_podcast_audio",
        "--mixed-audio",
        mixed.as_posix(),
        "--output-dir",
        out_dir.as_posix(),
        "--metadata-json",
        meta.as_posix(),
        "--backend",
        "mfcc",
    ]
    assert diarize.main() == 0
    assert (out_dir / "person1.wav").is_file()
    assert (out_dir / "person2.wav").is_file()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["backend"] == "mfcc"
    p1, _ = sf.read(out_dir / "person1.wav")
    p2, _ = sf.read(out_dir / "person2.wav")
    assert len(p1) == len(p2)



def test_main_pyannote_fallback(write_wav, tmp_path: Path, monkeypatch):
    mixed = write_wav(tmp_path / "mixed.wav", duration_sec=0.6)
    monkeypatch.setenv("HF_TOKEN", "fake")
    out_dir = tmp_path / "out2"
    sys.argv = [
        "diarize_podcast_audio",
        "--mixed-audio",
        mixed.as_posix(),
        "--output-dir",
        out_dir.as_posix(),
        "--backend",
        "pyannote",
    ]
    fake_pipe = MagicMock()
    fake_pipe.return_value.side_effect = RuntimeError("boom")
    with patch.dict("sys.modules", {"pyannote.audio": MagicMock(Pipeline=fake_pipe)}):
        with patch.object(diarize, "diarize_pyannote", side_effect=RuntimeError("boom")):
            assert diarize.main() == 0
    assert (out_dir / "person1.wav").exists()


def test_existing_audio_missing(tmp_path: Path):
    with pytest.raises(argparse.ArgumentTypeError):  # noqa: F821
        diarize.existing_audio((tmp_path / "nope.wav").as_posix())


# argparse for existing_audio test
import argparse  # noqa: E402


def test_align_speech_mask_trim_and_pad():
    mask = np.array([True, False, True])
    trimmed = diarize._align_speech_mask(mask, 2)
    assert len(trimmed) == 2
    padded = diarize._align_speech_mask(mask, 5)
    assert len(padded) == 5
    assert padded[4] is np.False_


def test_choose_backend_auto_pyannote_when_available(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    import types

    parent = types.ModuleType("pyannote")
    fake = types.ModuleType("pyannote.audio")
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake)
    assert diarize.choose_backend("auto") == "pyannote"


def test_diarize_mfcc_label_smoothing():
    sr = diarize.SAMPLE_RATE
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    y = np.zeros_like(t, dtype=np.float32)
    y[: sr] = 0.4 * np.sin(2 * np.pi * 200 * t[:sr])
    y[sr:] = 0.4 * np.sin(2 * np.pi * 400 * t[sr:])
    labels, mask = diarize.diarize_mfcc(y)
    assert len(labels) == len(mask)
    assert labels.max() <= 1


def test_diarize_pyannote_insufficient_speakers_falls_back():
    y = np.zeros(diarize.SAMPLE_RATE, dtype=np.float32)
    y[:4000] = 0.1
    path = Path("/tmp/fake.wav")

    class Turn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class Diar:
        def itertracks(self, yield_label=True):
            yield Turn(0.0, 0.2), None, "A"

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def __call__(self, _path):
            return Diar()

    import types

    fake = types.ModuleType("pyannote.audio")
    fake.Pipeline = Pipeline
    with patch.dict(sys.modules, {"pyannote.audio": fake}):
        labels, mask = diarize.diarize_pyannote(y, path)
    assert len(labels) > 0
