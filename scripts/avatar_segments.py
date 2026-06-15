#!/usr/bin/env python3
"""Segment count helpers for LongCat avatar clip rollout."""
from __future__ import annotations

import math
import wave
from pathlib import Path


def compute_num_segments(
    duration_sec: float,
    fps: int = 25,
    num_frames: int = 93,
    num_cond_frames: int = 13,
) -> int:
    first_clip = num_frames / fps
    if duration_sec <= first_clip:
        return 1
    per_segment = (num_frames - num_cond_frames) / fps
    remaining = duration_sec - first_clip
    extra = math.ceil(remaining / per_segment)
    return 1 + extra


def audio_duration_sec(path: Path) -> float | None:
    try:
        import librosa

        y, sr = librosa.load(path.as_posix(), sr=None, mono=True)
        return len(y) / float(sr)
    except Exception:
        if path.suffix.lower() != ".wav":
            return None
        try:
            with wave.open(path.as_posix(), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return None
