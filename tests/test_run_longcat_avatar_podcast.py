from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from avatar_segments import compute_num_segments

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


def make_manifest(tmp_path: Path, jobs: list[dict]) -> Path:
    image = write_asset(tmp_path / "assets" / "man.jpg")
    audio = write_asset(tmp_path / "assets" / "voice.m4a")
    manifest = tmp_path / "podcast.json"
    payload_jobs = []
    for job in jobs:
        payload_jobs.append(
            {
                "name": job["name"],
                "image": job.get("image", image.as_posix()),
                "audio": job.get("audio", audio.as_posix()),
                "prompt": job["prompt"],
            }
        )
    manifest.write_text(json.dumps({"jobs": payload_jobs}), encoding="utf-8")
    return manifest


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
    manifest = make_manifest(
        tmp_path,
        [
            {"name": "host", "prompt": "One"},
            {"name": "host", "prompt": "Two"},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate job name"):
        module.load_manifest(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            work_root=tmp_path / "work",
        )


def test_load_manifest_rejects_missing_manifest(tmp_path: Path) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="Manifest not found"):
        module.load_manifest(
            manifest_path=tmp_path / "missing.json",
            output_root=tmp_path / "outputs",
            work_root=tmp_path / "work",
        )


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    module = load_module()
    manifest = tmp_path / "bad.json"
    manifest.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        module.load_manifest(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            work_root=tmp_path / "work",
        )


def test_load_manifest_rejects_empty_jobs(tmp_path: Path) -> None:
    module = load_module()
    manifest = tmp_path / "empty.json"
    manifest.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        module.load_manifest(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            work_root=tmp_path / "work",
        )


def test_load_manifest_rejects_empty_prompt(tmp_path: Path) -> None:
    module = load_module()
    manifest = make_manifest(tmp_path, [{"name": "host", "prompt": "   "}])
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        module.load_manifest(
            manifest_path=manifest,
            output_root=tmp_path / "outputs",
            work_root=tmp_path / "work",
        )


def test_resolve_path_relative_to_manifest_dir(tmp_path: Path) -> None:
    module = load_module()
    image = write_asset(tmp_path / "faces" / "host.jpg")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    resolved = module.resolve_path(
        "../faces/host.jpg",
        base_dir=manifest_dir,
        exts=module.IMAGE_EXTS,
    )
    assert resolved == image.resolve()


def test_resolve_path_rejects_missing_and_bad_extension(tmp_path: Path) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="File not found"):
        module.resolve_path(
            "missing.jpg",
            base_dir=tmp_path,
            exts=module.IMAGE_EXTS,
        )
    bad = write_asset(tmp_path / "note.txt")
    with pytest.raises(ValueError, match="Unsupported file type"):
        module.resolve_path(
            bad.as_posix(),
            base_dir=tmp_path,
            exts=module.IMAGE_EXTS,
        )


def test_validate_name_rejects_empty_and_slashes() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="non-empty string"):
        module.validate_name("", 1)
    with pytest.raises(ValueError, match="path separators"):
        module.validate_name("foo/bar", 1)


def test_partition_jobs_round_robin(tmp_path: Path) -> None:
    module = load_module()
    manifest = make_manifest(
        tmp_path,
        [
            {"name": "a", "prompt": "A"},
            {"name": "b", "prompt": "B"},
            {"name": "c", "prompt": "C"},
            {"name": "d", "prompt": "D"},
            {"name": "e", "prompt": "E"},
        ],
    )
    jobs = module.load_manifest(
        manifest_path=manifest,
        output_root=tmp_path / "outputs",
        work_root=tmp_path / "work",
    )
    buckets = module.partition_jobs(jobs, 2)
    assert [job.name for job in buckets[0]] == ["a", "c", "e"]
    assert [job.name for job in buckets[1]] == ["b", "d"]


def test_write_input_json_creates_payload(
    tmp_path: Path, write_wav, monkeypatch
) -> None:
    module = load_module()
    image = write_asset(tmp_path / "host.jpg")
    audio = write_wav(tmp_path / "voice.wav", duration_sec=20.0)
    job = module.PodcastJob(
        name="host",
        image=image,
        audio=audio,
        prompt="Speak naturally.",
        input_json=tmp_path / "work" / "host.json",
        output_dir=tmp_path / "outputs" / "host",
        log_path=tmp_path / "logs" / "host.log",
    )
    module.write_input_json(job)
    payload = json.loads(job.input_json.read_text(encoding="utf-8"))
    assert payload["prompt"] == "Speak naturally."
    assert payload["cond_image"] == image.as_posix()
    assert payload["cond_audio"]["person1"] == audio.as_posix()
    assert payload["recommended_num_segments"] == compute_num_segments(20.0)


def test_run_job_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    image = write_asset(tmp_path / "host.jpg")
    audio = write_asset(tmp_path / "voice.m4a")
    job = module.PodcastJob(
        name="host",
        image=image,
        audio=audio,
        prompt="Speak.",
        input_json=tmp_path / "work" / "host.json",
        output_dir=tmp_path / "outputs" / "host",
        log_path=tmp_path / "logs" / "host.log",
    )
    run_script = tmp_path / "run.sh"
    run_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    run_script.chmod(0o755)
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, "env": kwargs.get("env", {})})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.run_job(job=job, gpu_id="1", run_script=run_script, cwd=tmp_path)
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert calls[0]["env"]["OUTPUT_DIR"] == job.output_dir.as_posix()
    assert job.log_path.is_file()

    def fake_run_fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module.subprocess, "run", fake_run_fail)
    with pytest.raises(RuntimeError, match="failed on GPU"):
        module.run_job(job=job, gpu_id="1", run_script=run_script, cwd=tmp_path)


def test_detect_gpu_ids_parses_nvidia_smi(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "shutil_which", lambda name: "/usr/bin/nvidia-smi")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="0\n1\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.detect_gpu_ids() == ["0", "1"]


def test_main_success_with_mocked_gpus(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    manifest = make_manifest(tmp_path, [{"name": "host", "prompt": "Hello."}])
    monkeypatch.setattr(module, "detect_gpu_ids", lambda: ["0"])
    monkeypatch.setattr(module, "run_worker", lambda **kwargs: None)
    sys.argv = [
        "run_longcat_avatar_podcast",
        "--manifest",
        manifest.as_posix(),
        "--work-root",
        (tmp_path / "work").as_posix(),
        "--output-root",
        (tmp_path / "outputs").as_posix(),
    ]
    assert module.main() == 0


def test_main_returns_1_on_worker_failure(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    manifest = make_manifest(tmp_path, [{"name": "host", "prompt": "Hello."}])

    def fail_worker(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "detect_gpu_ids", lambda: ["0"])
    monkeypatch.setattr(module, "run_worker", fail_worker)
    sys.argv = [
        "run_longcat_avatar_podcast",
        "--manifest",
        manifest.as_posix(),
        "--work-root",
        (tmp_path / "work").as_posix(),
        "--output-root",
        (tmp_path / "outputs").as_posix(),
    ]
    assert module.main() == 1


def test_resolve_worker_count_uses_gpu_limits() -> None:
    module = load_module()

    assert module.resolve_worker_count(None, job_count=2, gpu_count=4) == 2
    assert module.resolve_worker_count(4, job_count=2, gpu_count=4) == 2

    with pytest.raises(ValueError, match="No GPUs"):
        module.resolve_worker_count(None, job_count=2, gpu_count=0)

    with pytest.raises(ValueError, match="detected GPUs"):
        module.resolve_worker_count(5, job_count=2, gpu_count=4)
