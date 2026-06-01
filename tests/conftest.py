from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(module_name: str):
    path = SCRIPTS / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def write_wav(tmp_path: Path):
    def _write(
        path: Path,
        duration_sec: float = 1.0,
        sample_rate: int = 16000,
        left_tone: bool = True,
        right_tone: bool = False,
    ) -> Path:
        n = int(duration_sec * sample_rate)
        t = np.linspace(0, duration_sec, n, endpoint=False)
        signal = np.zeros(n, dtype=np.float32)
        if left_tone:
            signal += 0.3 * np.sin(2 * np.pi * 220 * t)
        if right_tone:
            signal += 0.3 * np.sin(2 * np.pi * 440 * t)
        pcm = np.clip(signal, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        with wave.open(path.as_posix(), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return path

    return _write


@pytest.fixture
def two_host_image(tmp_path: Path) -> Path:
    """Simple image with two high-contrast regions for face-detector fallback."""
    img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    pixels = img.load()
    for x in range(80, 240):
        for y in range(120, 320):
            pixels[x, y] = (220, 180, 160)
    for x in range(400, 560):
        for y in range(120, 320):
            pixels[x, y] = (200, 170, 150)
    path = tmp_path / "two_hosts.png"
    img.save(path)
    return path
