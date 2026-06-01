#!/usr/bin/env python3
"""Merge two per-speaker tracks into one mixed file (e.g. to simulate NotebookLM output)."""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person1-audio", required=True)
    parser.add_argument("--person2-audio", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a, _ = librosa.load(args.person1_audio, sr=SAMPLE_RATE, mono=True)
    b, _ = librosa.load(args.person2_audio, sr=SAMPLE_RATE, mono=True)
    length = max(len(a), len(b))
    mix = np.zeros(length, dtype=np.float32)
    if len(a) < length:
        a = np.pad(a, (0, length - len(a)))
    if len(b) < length:
        b = np.pad(b, (0, length - len(b)))
    mix[: len(a)] += a
    mix[: len(b)] += b
    peak = np.max(np.abs(mix)) or 1.0
    mix = 0.95 * mix / peak
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mix, SAMPLE_RATE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
