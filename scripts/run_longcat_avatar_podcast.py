#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from avatar_segments import audio_duration_sec, compute_num_segments

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class PodcastJob(NamedTuple):
    name: str
    image: Path
    audio: Path
    prompt: str
    input_json: Path
    output_dir: Path
    log_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default=(root / "LongCat-Video" / "outputs_podcast").as_posix())
    parser.add_argument("--work-root", default="/tmp/longcat-podcast")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-script", default=(root / "scripts" / "run_longcat_avatar_single.sh").as_posix())
    return parser.parse_args()


def resolve_path(raw_path: str, *, base_dir: Path, exts: set[str]) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() not in exts:
        allowed = ", ".join(sorted(exts))
        raise ValueError(f"Unsupported file type for {path.name}; expected one of: {allowed}")
    return path


def validate_name(raw_name: object, index: int) -> str:
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"Job {index} name must be a non-empty string")
    name = raw_name.strip()
    if "/" in name or "\\" in name:
        raise ValueError(f"Job name must not contain path separators: {name}")
    return name


def load_manifest(*, manifest_path: Path, output_root: Path, work_root: Path) -> list[PodcastJob]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {exc}") from exc
    jobs_payload = payload.get("jobs")
    if not isinstance(jobs_payload, list) or not jobs_payload:
        raise ValueError("Manifest jobs must be a non-empty list")

    jobs: list[PodcastJob] = []
    seen_names: set[str] = set()
    manifest_dir = manifest_path.parent
    log_root = work_root / "logs"
    for index, job_payload in enumerate(jobs_payload, start=1):
        if not isinstance(job_payload, dict):
            raise ValueError(f"Job {index} must be an object")
        name = validate_name(job_payload.get("name"), index)
        if name in seen_names:
            raise ValueError(f"Duplicate job name: {name}")
        seen_names.add(name)

        prompt = job_payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Job {name} prompt must be a non-empty string")

        image = resolve_path(job_payload.get("image", ""), base_dir=manifest_dir, exts=IMAGE_EXTS)
        audio = resolve_path(job_payload.get("audio", ""), base_dir=manifest_dir, exts=AUDIO_EXTS)
        jobs.append(
            PodcastJob(
                name=name,
                image=image,
                audio=audio,
                prompt=prompt.strip(),
                input_json=work_root / f"{name}.json",
                output_dir=output_root / name,
                log_path=log_root / f"{name}.log",
            )
        )
    return jobs


def detect_gpu_ids() -> list[str]:
    if not shutil_which("nvidia-smi"):
        return []
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.as_posix()
    return None


def resolve_worker_count(requested_workers: int | None, *, job_count: int, gpu_count: int) -> int:
    if gpu_count < 1:
        raise ValueError("No GPUs detected")
    if requested_workers is None:
        return min(job_count, gpu_count)
    if requested_workers < 1:
        raise ValueError("workers must be at least 1")
    if requested_workers > gpu_count:
        raise ValueError(f"Requested workers ({requested_workers}) exceed detected GPUs ({gpu_count})")
    return min(requested_workers, job_count)


def write_input_json(job: PodcastJob) -> None:
    job.input_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": job.prompt,
        "cond_image": job.image.as_posix(),
        "cond_audio": {"person1": job.audio.as_posix()},
    }
    duration_sec = audio_duration_sec(job.audio)
    if duration_sec is not None:
        payload["recommended_num_segments"] = compute_num_segments(float(duration_sec))
    job.input_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def partition_jobs(jobs: list[PodcastJob], worker_count: int) -> list[list[PodcastJob]]:
    buckets = [[] for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        buckets[index % worker_count].append(job)
    return buckets


def run_job(*, job: PodcastJob, gpu_id: str, run_script: Path, cwd: Path) -> None:
    write_input_json(job)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["OUTPUT_DIR"] = job.output_dir.as_posix()
    env.setdefault("NPROC_PER_NODE", "1")
    env.setdefault("CONTEXT_PARALLEL_SIZE", "1")

    command = ["bash", run_script.as_posix(), job.input_json.as_posix()]
    print(f"[{job.name}] starting on GPU {gpu_id}; log={job.log_path}", flush=True)
    with job.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Running on GPU {gpu_id}: {' '.join(command)}\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Job {job.name} failed on GPU {gpu_id}; see {job.log_path}")
    print(f"[{job.name}] finished; output={job.output_dir}", flush=True)


def run_worker(*, gpu_id: str, jobs: list[PodcastJob], run_script: Path, cwd: Path) -> None:
    for job in jobs:
        run_job(job=job, gpu_id=gpu_id, run_script=run_script, cwd=cwd)


def main() -> int:
    args = parse_args()
    cwd = repo_root()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    run_script = Path(args.run_script).expanduser().resolve()
    requested_workers = args.workers
    if requested_workers is None and os.environ.get("PODCAST_WORKERS"):
        requested_workers = int(os.environ["PODCAST_WORKERS"])

    jobs = load_manifest(manifest_path=manifest_path, output_root=output_root, work_root=work_root)
    gpu_ids = detect_gpu_ids()
    worker_count = resolve_worker_count(requested_workers, job_count=len(jobs), gpu_count=len(gpu_ids))
    selected_gpu_ids = gpu_ids[:worker_count]
    job_buckets = partition_jobs(jobs, worker_count)

    print(f"manifest: {manifest_path}")
    print(f"jobs: {len(jobs)}")
    print(f"gpus: {', '.join(gpu_ids)}")
    print(f"workers: {worker_count}")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_worker, gpu_id=gpu_id, jobs=bucket, run_script=run_script, cwd=cwd)
            for gpu_id, bucket in zip(selected_gpu_ids, job_buckets)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))

    if failures:
        for failure in failures:
            print(f"error: {failure}")
        return 1

    print("status: succeeded")
    for job in jobs:
        print(f"output_dir: {job.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
