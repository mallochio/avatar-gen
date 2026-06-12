# avatar-gen

Turn a **portrait image** and **audio clip** into a talking-head video using [LongCat-Video-Avatar-1.5](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5).

The default path runs on a **GCP spot H100** via SkyPilot. You place files locally, run one script, and get an MP4 back in `outputs/`.

Upstream model code lives in the [`LongCat-Video/`](LongCat-Video/) git submodule.

## What you need

- macOS or Linux laptop (inference runs in the cloud; you do not need a local GPU)
- [Google Cloud](https://cloud.google.com/) project with **H100 / A3 quota** in `europe-west1`
- `gcloud` CLI authenticated: `gcloud auth application-default login`
- Git with submodules

First run downloads model weights on the VM (~30–40 minutes). Later runs are faster if you keep the cluster warm (see below).

## Quick start (cloud)

```bash
git clone --recurse-submodules git@github.com:mallochio/avatar-gen.git
cd avatar-gen

# 1. Add your files
cp /path/to/face.png inputs/portrait.png
cp /path/to/voice.mp3 inputs/audio.mp3
# optional:
# echo "A person presents quarterly results in a modern office." > inputs/prompt.txt

# 2. Generate
bash scripts/generate_avatar.sh

# 3. Open the result
open outputs/avatar.mp4    # macOS
# xdg-open outputs/avatar.mp4   # Linux
```

`generate_avatar.sh` installs SkyPilot on first run (`scripts/setup_skypilot.sh`), provisions a spot `a3-highgpu-*` VM in `europe-west1`, runs inference, syncs `outputs/` back to your machine, and tears the cluster down.

### Input files

| File | Required | Formats |
|------|----------|---------|
| `inputs/portrait.png` | yes | `.png`, `.jpg`, `.jpeg`, `.webp` |
| `inputs/audio.mp3` | yes | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` |
| `inputs/prompt.txt` | no | plain text scene description |

### Optional flags

```bash
NUM_GPUS=8 bash scripts/generate_avatar.sh   # more GPUs for long audio (2, 4, or 8)
USE_SPOT=0 bash scripts/generate_avatar.sh   # on-demand instead of spot
```

Multi-GPU helps mainly with longer audio via context parallelism. The default `NUM_GPUS=1` is right for most clips.

To reuse a warm cluster (skip re-provisioning), remove `--down` from the launch command in `scripts/generate_avatar.sh` and run `sky down avatar-gen -y` when finished.

## Local Linux GPU (optional)

If you have an NVIDIA GPU with CUDA 12.4+:

```bash
bash scripts/setup_uv_env.sh
bash scripts/download_longcat_weights.sh

LongCat-Video/.venv/bin/python scripts/prepare_longcat_avatar_input.py \
  --image inputs/portrait.png \
  --audio inputs/audio.mp3 \
  --prompt "Your scene." \
  --output-json /tmp/input.json

PATH="LongCat-Video/.venv/bin:$PATH" OUTPUT_DIR=outputs \
  bash scripts/run_longcat_avatar_single.sh /tmp/input.json
cp outputs/ai2v_demo_1.mp4 outputs/avatar.mp4
```

## Experimental: two-host podcast

The podcast pipeline (diarize mixed audio, two faces in one frame) is **experimental and not validated end-to-end**. See [`docs/EXPERIMENTAL_PODCAST.md`](docs/EXPERIMENTAL_PODCAST.md).

## Project layout

```
avatar-gen/
  inputs/           ← your portrait + audio (+ optional prompt.txt)
  outputs/          ← avatar.mp4 appears here after a cloud run
  skypilot/avatar.yaml
  scripts/          ← generate_avatar.sh and helpers
  LongCat-Video/    ← upstream submodule (weights downloaded on first run)
```

## Troubleshooting

**Submodule empty** — `git submodule update --init --recursive`

**Missing inputs** — script prints expected paths under `inputs/`

**Quota / provisioning errors** — check H100 availability:

```bash
gcloud compute accelerator-types list --filter="name~h100 AND zone~europe-west1"
```

Request a quota increase in Cloud Console if needed.

**Spot preemption** — retry, or `USE_SPOT=0 bash scripts/generate_avatar.sh`

**`torchrun: not found` (local runs)** — `PATH="LongCat-Video/.venv/bin:$PATH"`

Maintainer notes: [`SPEC.md`](SPEC.md)

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest --cov=scripts --cov-fail-under=80
```

## License

LongCat-Video upstream is MIT. See [`LongCat-Video/LICENSE`](LongCat-Video/LICENSE).
