#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def default_roots() -> tuple[Path, Path]:
    overlay_root = Path(__file__).resolve().parents[1]
    longcat_root = Path(os.environ.get("LONGCAT_ROOT", overlay_root / "LongCat-Video")).expanduser().resolve()
    return overlay_root, longcat_root


def parse_args() -> argparse.Namespace:
    overlay_root, longcat_root = default_roots()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=longcat_root.as_posix())
    parser.add_argument("--overlay-root", default=overlay_root.as_posix())
    parser.add_argument("--weights-root", default=(longcat_root / "weights").as_posix())
    parser.add_argument(
        "--input-json",
        default=(overlay_root / "assets/avatar/single_example_uv.json").as_posix(),
    )
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p"])
    parser.add_argument("--num-segments", type=int, default=1)
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--no-distill", action="store_true")
    return parser.parse_args()


def required_paths(weights_root: Path, use_int8: bool, use_distill: bool) -> list[tuple[Path, str]]:
    items = [
        (weights_root / "LongCat-Video/tokenizer", "LongCat tokenizer"),
        (weights_root / "LongCat-Video/text_encoder", "LongCat text encoder"),
        (weights_root / "LongCat-Video/vae", "LongCat VAE"),
        (weights_root / "LongCat-Video-Avatar-1.5/scheduler", "Avatar 1.5 scheduler"),
        (weights_root / "LongCat-Video-Avatar-1.5/vocal_separator/Kim_Vocal_2.onnx", "Vocal separator ONNX"),
        (weights_root / "LongCat-Video-Avatar-1.5/whisper-large-v3/config.json", "Whisper config"),
        (weights_root / "LongCat-Video-Avatar-1.5/whisper-large-v3/preprocessor_config.json", "Whisper preprocessor"),
        (weights_root / "LongCat-Video-Avatar-1.5/whisper-large-v3/model.safetensors", "Whisper weights"),
    ]
    if use_int8:
        items.extend(
            [
                (weights_root / "LongCat-Video-Avatar-1.5/base_model_int8/config.json", "INT8 DiT config"),
                (weights_root / "LongCat-Video-Avatar-1.5/base_model_int8/quantized_model.safetensors.index.json", "INT8 DiT index"),
            ]
        )
    else:
        items.extend(
            [
                (weights_root / "LongCat-Video-Avatar-1.5/base_model/config.json", "FP16 DiT config"),
                (weights_root / "LongCat-Video-Avatar-1.5/base_model/diffusion_pytorch_model.safetensors.index.json", "FP16 DiT index"),
            ]
        )
    if use_distill:
        items.append((weights_root / "LongCat-Video-Avatar-1.5/lora/dmd_lora.safetensors", "DMD distill LoRA"))
    return items


def resolve_input_path(raw_path: str, overlay_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (overlay_root / path).resolve()


def load_input_json(path: Path, overlay_root: Path) -> tuple[list[str], dict[str, object] | None]:
    issues: list[str] = []
    if not path.is_file():
        return [f"Input JSON missing: {path}"], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Input JSON invalid: {exc}"], None
    if not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
        issues.append("prompt must be a non-empty string")
    image_path = payload.get("cond_image")
    if not isinstance(image_path, str):
        issues.append(f"cond_image missing or not found: {image_path}")
    else:
        resolved_image = resolve_input_path(image_path, overlay_root)
        if not resolved_image.is_file():
            issues.append(f"cond_image missing or not found: {resolved_image}")
    cond_audio = payload.get("cond_audio")
    person1 = cond_audio.get("person1") if isinstance(cond_audio, dict) else None
    if not isinstance(person1, str):
        issues.append(f"cond_audio.person1 missing or not found: {person1}")
    else:
        resolved_audio = resolve_input_path(person1, overlay_root)
        if not resolved_audio.is_file():
            issues.append(f"cond_audio.person1 missing or not found: {resolved_audio}")
    person2 = cond_audio.get("person2") if isinstance(cond_audio, dict) else None
    if isinstance(person2, str):
        resolved_audio2 = resolve_input_path(person2, overlay_root)
        if not resolved_audio2.is_file():
            issues.append(f"cond_audio.person2 missing or not found: {resolved_audio2}")
    return issues, payload


def detect_gpus() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_command(
    repo_root: Path,
    input_json: Path,
    resolution: str,
    num_segments: int,
    use_int8: bool,
    use_distill: bool,
    gpu_count: int,
) -> str:
    nproc = 2 if gpu_count >= 2 else 1
    flags = [
        f"bash scripts/run_longcat_avatar_single.sh {input_json.as_posix()}",
        f"# env: OUTPUT_DIR={repo_root.as_posix()}/outputs_avatar_single",
        f"# upstream cwd: {repo_root.as_posix()}",
        f"# resolution={resolution} num_segments={num_segments} nproc={nproc}",
    ]
    if use_distill:
        flags.append("# USE_DISTILL=1")
    if use_int8:
        flags.append("# USE_INT8=1")
    return " ".join(flags)


def main() -> int:
    args = parse_args()
    overlay_root = Path(args.overlay_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    weights_root = Path(args.weights_root).expanduser().resolve()
    input_json = Path(args.input_json).expanduser().resolve()
    use_int8 = not args.no_int8
    use_distill = not args.no_distill

    path_issues = [
        f"Missing {label}: {path}"
        for path, label in required_paths(weights_root, use_int8, use_distill)
        if not path.exists()
    ]
    input_issues, payload = load_input_json(input_json, overlay_root)

    gpus = detect_gpus()
    runtime_supported = platform.system() == "Linux" and bool(gpus)
    ffmpeg_path = shutil.which("ffmpeg") or "missing"

    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"longcat_root: {repo_root}")
    print(f"overlay_root: {overlay_root}")
    print(f"ffmpeg: {ffmpeg_path}")
    print(f"gpus: {len(gpus)}")
    for gpu in gpus:
        print(f"gpu_name: {gpu}")
    print(f"runtime_supported: {'yes' if runtime_supported else 'no'}")

    if payload is not None:
        print(f"input_prompt_chars: {len(payload['prompt'].strip())}")
        print(f"input_image: {payload['cond_image']}")
        print(f"input_audio: {payload['cond_audio']['person1']}")

    if path_issues or input_issues:
        print("status: missing_requirements")
        for issue in path_issues + input_issues:
            print(f"issue: {issue}")
        return 1

    command = build_command(repo_root, input_json, args.resolution, args.num_segments, use_int8, use_distill, len(gpus))
    print("status: ready_for_supported_host")
    print(f"recommended_command: {command}")
    if not runtime_supported:
        print(
            "warning: upstream LongCat avatar inference is hard-wired to Linux + NVIDIA CUDA + NCCL; "
            "this host can validate assets and weights but cannot run generation as-is"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
