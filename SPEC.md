# avatar-gen — maintainer notes

User-facing guide: [`README.md`](README.md).

## Cloud defaults

- Provider: GCP
- Region: `europe-west1`
- Instance: `a3-highgpu-{1,2,4,8}g` (spot by default)
- SkyPilot task: [`skypilot/avatar.yaml`](skypilot/avatar.yaml)
- Launch: `bash scripts/generate_avatar.sh`

## SkyPilot config

Use the active `gcloud` project:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

SkyPilot: `uv sync --group cloud --prerelease allow` then `uv run --group cloud sky ...`.

## Quota checks

```bash
gcloud compute accelerator-types list --filter="name~h100 AND zone~europe-west1"
gcloud compute regions describe europe-west1 --format=json \
  | jq -r '.quotas[] | select(.metric|test("GPU|A100|V100")) | "\(.metric) usage=\(.usage) limit=\(.limit)"'
```

H100 quota may not appear as a legacy `NVIDIA_H100_*` row; request **NVIDIA H100 GPUs** or **A3** quota in Cloud Console if launch fails.

## Output storage

`AVATAR_OUTPUT_BUCKET` must point at a writable GCS bucket:

```bash
export AVATAR_OUTPUT_BUCKET=gs://YOUR_AVATAR_BUCKET
```

SkyPilot mounts the bucket at `/outputs`, writes each run under `/outputs/avatar-gen/run-${SKYPILOT_TASK_ID}`, copies the latest result to `/outputs/avatar-gen/latest/avatar.mp4`, then `scripts/generate_avatar.sh` downloads it to `outputs/avatar.mp4`.

## Blocker history (archive)

| Issue | Resolution |
|-------|------------|
| Azure NCC: no GPU on fresh VM | `setup_ncc_gpu.sh` (Azure-only; Azure paths removed) |
| Azure SkyPilot patches / catalog | Deprecated; GCP is primary |
| Podcast SkyPilot runs Jun 2026 | Failed on driver setup; podcast demoted to experimental local-only |
| Mac SkyPilot BrokenProcessPool | Bypassed historically via Azure API VM; GCP launch from laptop is fine |

## Experimental podcast

See [`docs/EXPERIMENTAL_PODCAST.md`](docs/EXPERIMENTAL_PODCAST.md). Scripts remain under `scripts/` but are not part of the supported cloud workflow.
