# Experimental: two-host podcast pipeline

**Status: experimental.** This path has not been validated end-to-end in this repo. There is no SkyPilot YAML or cloud launcher for podcast generation.

## What it tries to do

Given a seed image with two people and mixed audio (or separate per-speaker tracks), the pipeline:

1. Diarizes audio (`scripts/diarize_podcast_audio.py`)
2. Detects host bounding boxes in the seed frame (`scripts/detect_podcast_hosts.py`)
3. Builds a multi-person input JSON (`scripts/prepare_longcat_avatar_multi_input.py`)
4. Runs multi-person inference (`scripts/run_longcat_avatar_multi.sh`)

An alternate path runs parallel single-host clips from a manifest (`scripts/run_longcat_avatar_podcast.py`).

## Local usage only

Requires a Linux machine with NVIDIA GPU, CUDA, and weights already installed (`scripts/setup_uv_env.sh`, `scripts/download_longcat_weights.sh`).

Integrated pipeline:

```bash
bash scripts/run_podcast_pipeline.sh \
  --seed-image /path/to/seed.png \
  --mixed-audio /path/to/mixed.mp3 \
  --prompt "Two hosts in a podcast studio."
```

Manifest runner:

```bash
LongCat-Video/.venv/bin/python scripts/run_longcat_avatar_podcast.py \
  --manifest assets/podcast/podcast_jobs.json \
  --output-root LongCat-Video/outputs_podcast
```

Example assets: `assets/podcast/podcast_jobs.json`, `assets/avatar/podcast/example_podcast.json`.

## Known gaps

- No cloud orchestration for podcast jobs
- Azure/GCP podcast SkyPilot YAMLs were removed; do not expect them to return without new work
- Base model (50-NFE, no int8/distill) is slower and heavier than the default single-host avatar path

For production talking-head clips from one face + one audio track, use `bash scripts/generate_avatar.sh` instead.
