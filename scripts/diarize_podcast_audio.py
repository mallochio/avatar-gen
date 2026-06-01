#!/usr/bin/env python3
"""Separate a mixed two-host podcast track into per-speaker WAV files.

Uses frame-level voice-activity detection plus spectral clustering on MFCC
features (Active Speaker-style routing for avatar conditioning). When
``pyannote.audio`` is installed and ``HF_TOKEN`` is set, a pyannote
diarization pipeline is preferred.

Each output track is the same duration as the input; non-speaking regions are
zeroed so the listening host receives the model's silent-audio condition.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def existing_audio(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Audio file not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-audio", required=True, type=existing_audio)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--person1-label",
        default="person1",
        help="Output stem for left / Host A track (default: person1)",
    )
    parser.add_argument(
        "--person2-label",
        default="person2",
        help="Output stem for right / Host B track (default: person2)",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional path to write diarization metadata JSON",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pyannote", "mfcc"],
        default="auto",
        help="Diarization backend (default: auto)",
    )
    return parser.parse_args()


def _frame_energy_mask(y: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mask = np.zeros(int(np.ceil(len(y) / HOP_LENGTH)), dtype=bool)
    for start, end in intervals:
        f0 = start // HOP_LENGTH
        f1 = int(np.ceil(end / HOP_LENGTH))
        mask[f0:f1] = True
    return mask


def _mfcc_features(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([mfcc, delta], axis=0).T


def diarize_mfcc(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame speaker id (0 or 1) aligned to audio hops."""
    speech_mask = _frame_energy_mask(y)
    feats = _mfcc_features(y)
    active = feats[speech_mask]
    if active.shape[0] < 4:
        half = len(y) // 2
        labels = np.zeros(len(feats), dtype=np.int64)
        mid_frame = (half // HOP_LENGTH)
        labels[mid_frame:] = 1
        return labels, speech_mask

    scaled = StandardScaler().fit_transform(active)
    clusterer = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
    active_labels = clusterer.fit_predict(scaled)

    labels = np.zeros(len(feats), dtype=np.int64)
    labels[speech_mask] = active_labels

    # Smooth short flips
    for _ in range(2):
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] != labels[i]:
                labels[i] = labels[i - 1]
    return labels, speech_mask


def diarize_pyannote(y: np.ndarray, wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from pyannote.audio import Pipeline  # type: ignore

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    diarization = pipeline(str(wav_path))

    frame_count = int(np.ceil(len(y) / HOP_LENGTH))
    labels = np.zeros(frame_count, dtype=np.int64)
    speech_mask = np.zeros(frame_count, dtype=bool)

    speakers = sorted({label for _, _, label in diarization.itertracks(yield_label=True)})
    if len(speakers) < 2:
        return diarize_mfcc(y)

    spk_to_id = {spk: idx for idx, spk in enumerate(speakers[:2])}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in spk_to_id:
            continue
        start_f = int(turn.start * SAMPLE_RATE / HOP_LENGTH)
        end_f = int(turn.end * SAMPLE_RATE / HOP_LENGTH)
        labels[start_f:end_f] = spk_to_id[speaker]
        speech_mask[start_f:end_f] = True
    return labels, speech_mask


def labels_to_tracks(y: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    person1 = np.zeros_like(y, dtype=np.float32)
    person2 = np.zeros_like(y, dtype=np.float32)
    for frame_idx, spk in enumerate(labels):
        start = frame_idx * HOP_LENGTH
        end = min(len(y), start + HOP_LENGTH)
        if end <= start:
            continue
        chunk = y[start:end]
        if spk == 0:
            person1[start:end] = chunk
        else:
            person2[start:end] = chunk
    return person1, person2


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pyannote.audio  # noqa: F401

        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return "pyannote"
    except ImportError:
        pass
    return "mfcc"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    y, _ = librosa.load(args.mixed_audio.as_posix(), sr=SAMPLE_RATE, mono=True)
    backend = choose_backend(args.backend)
    if backend == "pyannote":
        try:
            labels, speech_mask = diarize_pyannote(y, args.mixed_audio)
        except Exception as exc:  # noqa: BLE001
            print(f"pyannote diarization failed ({exc}); falling back to mfcc clustering")
            labels, speech_mask = diarize_mfcc(y)
            backend = "mfcc"
    else:
        labels, speech_mask = diarize_mfcc(y)

    person1, person2 = labels_to_tracks(y, labels)

    p1_path = out_dir / f"{args.person1_label}.wav"
    p2_path = out_dir / f"{args.person2_label}.wav"
    sf.write(p1_path, person1, SAMPLE_RATE)
    sf.write(p2_path, person2, SAMPLE_RATE)

    meta = {
        "mixed_audio": args.mixed_audio.as_posix(),
        "sample_rate": SAMPLE_RATE,
        "backend": backend,
        "person1_audio": p1_path.as_posix(),
        "person2_audio": p2_path.as_posix(),
        "duration_sec": float(len(y) / SAMPLE_RATE),
        "silent_condition": (
            "Non-speaking regions are zeroed per track so the listening host "
            "receives silent audio (LongCat multi-person routing)."
        ),
    }
    if args.metadata_json:
        meta_path = Path(args.metadata_json).expanduser().resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(p1_path)
    print(p2_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
