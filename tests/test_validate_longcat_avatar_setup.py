from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import load_script

validate = load_script("validate_longcat_avatar_setup")


def test_default_roots(repo_root: Path):
    overlay, longcat = validate.default_roots()
    assert overlay == repo_root
    assert longcat.name == "LongCat-Video"


def test_resolve_input_path_relative_and_absolute(repo_root: Path, tmp_path: Path):
    rel = validate.resolve_input_path("assets/avatar/custom/random_person.png", repo_root)
    assert rel.is_absolute()
    abs_path = validate.resolve_input_path(tmp_path.as_posix(), repo_root)
    assert abs_path == tmp_path.resolve()


def test_required_paths_fp16_vs_int8(tmp_path: Path):
    weights = tmp_path / "weights"
    fp16 = validate.required_paths(weights, use_int8=False, use_distill=False)
    int8 = validate.required_paths(weights, use_int8=True, use_distill=False)
    assert any("base_model/" in p[0].as_posix() for p in fp16)
    assert any("base_model_int8" in p[0].as_posix() for p in int8)
    distill = validate.required_paths(weights, use_int8=False, use_distill=True)
    assert any("dmd_lora" in p[0].as_posix() for p in distill)


def test_load_input_json_missing_file(repo_root: Path):
    issues, payload = validate.load_input_json(repo_root / "nope.json", repo_root)
    assert issues
    assert payload is None


def test_load_input_json_invalid_and_valid(
    repo_root: Path, two_host_image: Path, write_wav, tmp_path: Path
):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    issues, _ = validate.load_input_json(bad, repo_root)
    assert any("invalid" in i.lower() for i in issues)

    audio = write_wav(tmp_path / "a.wav")
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "prompt": "Hi",
                "cond_image": two_host_image.as_posix(),
                "cond_audio": {"person1": audio.as_posix()},
            }
        ),
        encoding="utf-8",
    )
    issues, payload = validate.load_input_json(good, repo_root)
    assert not issues
    assert payload is not None


def test_load_input_json_person2_optional(repo_root: Path, two_host_image: Path, write_wav, tmp_path: Path):
    a = write_wav(tmp_path / "a.wav")
    b = write_wav(tmp_path / "b.wav")
    path = tmp_path / "multi.json"
    path.write_text(
        json.dumps(
            {
                "prompt": "Hi",
                "cond_image": two_host_image.as_posix(),
                "cond_audio": {"person1": a.as_posix(), "person2": (tmp_path / "missing.wav").as_posix()},
            }
        ),
        encoding="utf-8",
    )
    issues, _ = validate.load_input_json(path, repo_root)
    assert any("person2" in i for i in issues)


def test_detect_gpus_no_smi(monkeypatch):
    monkeypatch.setattr(validate.shutil, "which", lambda _: None)
    assert validate.detect_gpus() == []


def test_detect_gpus_parses_output(monkeypatch):
    monkeypatch.setattr(validate.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    class Result:
        returncode = 0
        stdout = "GPU A\nGPU B\n"

    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: Result())
    assert validate.detect_gpus() == ["GPU A", "GPU B"]


def test_build_command_flags():
    cmd = validate.build_command(
        Path("/repo"),
        Path("/in.json"),
        resolution="720p",
        num_segments=3,
        use_int8=True,
        use_distill=True,
        gpu_count=2,
    )
    assert "run_longcat_avatar_single.sh" in cmd
    assert "USE_DISTILL=1" in cmd
    assert "USE_INT8=1" in cmd


def test_main_missing_weights(repo_root: Path, two_host_image: Path, write_wav, tmp_path: Path, capsys):
    audio = write_wav(tmp_path / "a.wav")
    input_json = tmp_path / "in.json"
    input_json.write_text(
        json.dumps(
            {
                "prompt": "Test",
                "cond_image": two_host_image.as_posix(),
                "cond_audio": {"person1": audio.as_posix()},
            }
        ),
        encoding="utf-8",
    )
    sys.argv = [
        "validate_longcat_avatar_setup",
        "--overlay-root",
        repo_root.as_posix(),
        "--repo-root",
        (repo_root / "LongCat-Video").as_posix(),
        "--weights-root",
        (tmp_path / "empty_weights").as_posix(),
        "--input-json",
        input_json.as_posix(),
        "--no-int8",
        "--no-distill",
    ]
    assert validate.main() == 1
    out = capsys.readouterr().out
    assert "missing_requirements" in out


def test_main_ready_when_weights_exist(
    repo_root: Path, two_host_image: Path, write_wav, tmp_path: Path, capsys
):
    audio = write_wav(tmp_path / "a.wav")
    input_json = tmp_path / "in.json"
    input_json.write_text(
        json.dumps(
            {
                "prompt": "Test",
                "cond_image": two_host_image.as_posix(),
                "cond_audio": {"person1": audio.as_posix()},
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "weights"
    for rel in [
        "LongCat-Video/tokenizer",
        "LongCat-Video/text_encoder",
        "LongCat-Video/vae",
        "LongCat-Video-Avatar-1.5/scheduler",
        "LongCat-Video-Avatar-1.5/vocal_separator/Kim_Vocal_2.onnx",
        "LongCat-Video-Avatar-1.5/whisper-large-v3/config.json",
        "LongCat-Video-Avatar-1.5/whisper-large-v3/preprocessor_config.json",
        "LongCat-Video-Avatar-1.5/whisper-large-v3/model.safetensors",
        "LongCat-Video-Avatar-1.5/base_model/config.json",
        "LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json",
    ]:
        p = weights / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    sys.argv = [
        "validate_longcat_avatar_setup",
        "--overlay-root",
        repo_root.as_posix(),
        "--repo-root",
        (repo_root / "LongCat-Video").as_posix(),
        "--weights-root",
        weights.as_posix(),
        "--input-json",
        input_json.as_posix(),
        "--no-int8",
        "--no-distill",
    ]
    with patch.object(validate, "detect_gpus", return_value=["Fake GPU"]):
        assert validate.main() == 0
    out = capsys.readouterr().out
    assert "ready_for_supported_host" in out
