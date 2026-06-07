from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_longcat_avatar_podcast.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_longcat_avatar_podcast", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_asset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_load_manifest_builds_jobs(tmp_path: Path) -> None:
    module = load_module()
    image = write_asset(tmp_path / "assets" / "man.jpg")
    audio = write_asset(tmp_path / "assets" / "voice.m4a")
    manifest = tmp_path / "podcast.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "man",
                        "image": image.as_posix(),
                        "audio": audio.as_posix(),
                        "prompt": "Podcast host speaks naturally.",
                    },
                    {
                        "name": "woman",
                        "image": image.as_posix(),
                        "audio": audio.as_posix(),
                        "prompt": "Podcast co-host speaks naturally.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = module.load_manifest(
        manifest_path=manifest,
        output_root=tmp_path / "outputs",
        work_root=tmp_path / "work",
    )

    assert [job.name for job in jobs] == ["man", "woman"]
    assert jobs[0].output_dir == tmp_path / "outputs" / "man"
    assert jobs[1].input_json == tmp_path / "work" / "woman.json"
    assert jobs[0].prompt == "Podcast host speaks naturally."


def test_load_manifest_rejects_duplicate_names(tmp_path: Path) -> None:
    module = load_module()
    image = write_asset(tmp_path / "assets" / "man.jpg")
    audio = write_asset(tmp_path / "assets" / "voice.m4a")
    manifest = tmp_path / "podcast.json"
    manifest.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "host",
                        "image": image.as_posix(),
                        "audio": audio.as_posix(),
                        "prompt": "One",
                    },
                    {
                        "name": "host",
                        "image": image.as_posix(),
                        "audio": audio.as_posix(),
                        "prompt": "Two",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate job name"):
        module.load_manifest(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            work_root=tmp_path / "work",
        )


def test_resolve_worker_count_uses_gpu_limits() -> None:
    module = load_module()

    assert module.resolve_worker_count(None, job_count=2, gpu_count=4) == 2
    assert module.resolve_worker_count(4, job_count=2, gpu_count=4) == 2

    with pytest.raises(ValueError, match="No GPUs"):
        module.resolve_worker_count(None, job_count=2, gpu_count=0)

    with pytest.raises(ValueError, match="detected GPUs"):
        module.resolve_worker_count(5, job_count=2, gpu_count=4)
