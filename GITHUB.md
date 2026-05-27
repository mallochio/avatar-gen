# GitHub repo

This repo is live at **https://github.com/mallochio/avatar-gen** (private).

## Structure

```
avatar-gen/                          # this repo
  README.md                          # user run guide
  SPEC.md                            # Azure ops spec
  scripts/                           # wrapper scripts
  skypilot/                          # SkyPilot YAML
  assets/avatar/custom/              # demo inputs
  LongCat-Video/                     # git submodule → meituan-longcat/LongCat-Video
```

## Submodule workflow

Initialize after clone:

```bash
git submodule update --init --recursive
```

Update upstream:

```bash
cd LongCat-Video
git fetch origin
git checkout main
git pull
cd ..
git add LongCat-Video
git commit -m "chore: bump LongCat-Video submodule"
```

Pin to a tested commit before production runs.

## What stays out of git

- `LongCat-Video/weights/` — download via `scripts/download_longcat_weights.sh`
- `LongCat-Video/outputs_avatar_single/` — generated videos
- `.venv/`, `.sky-venv/` — local Python envs
- `~/.sky/config.yaml` — account-specific SkyPilot config
- VM-only SkyPilot patches — document in SPEC.md

## SkyPilot patches (control-plane VM)

Re-apply after SkyPilot upgrade on the API server VM:

1. `network.SubResource` fix in `sky/provision/azure/instance.py`
2. ConfidentialVM image block for NCC SKUs in `_create_vm`
3. `Standard_NCC40ads_H100_v5` row in `~/.sky/catalogs/v8/azure/vms.csv` for westeurope

See [`SPEC.md`](SPEC.md) for details.
