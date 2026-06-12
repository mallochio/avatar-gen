# GitHub repo

Live at **https://github.com/mallochio/avatar-gen** (private).

## Structure

```
avatar-gen/
  README.md              # user guide
  SPEC.md                # maintainer notes
  docs/EXPERIMENTAL_PODCAST.md
  inputs/                # user portrait + audio (gitignored contents)
  outputs/               # generated avatar.mp4 (gitignored)
  scripts/               # generate_avatar.sh + helpers
  skypilot/avatar.yaml   # single SkyPilot task
  LongCat-Video/         # submodule
```

## Submodule workflow

```bash
git submodule update --init --recursive
```

## What stays out of git

- `LongCat-Video/weights/` — downloaded on VM or locally
- `inputs/*` except `inputs/.gitkeep`
- `outputs/`
- `.venv/`, `.sky-venv/`, `.sky/`
- `~/.sky/config.yaml`

## SkyPilot

```bash
bash scripts/setup_skypilot.sh
bash scripts/generate_avatar.sh
```
