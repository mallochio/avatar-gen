# avatar-gen

Run **[LongCat-Video-Avatar-1.5](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5)** on your own image + audio: locally on an NVIDIA GPU, or on Azure via SkyPilot / manual VM launch.

Upstream model code: [`LongCat-Video/`](LongCat-Video/) (git submodule).

Ops history and Azure specifics: [`SPEC.md`](SPEC.md).

## Clone

```bash
git clone --recurse-submodules git@github.com:mallochio/avatar-gen.git
cd avatar-gen
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest --cov=scripts --cov-fail-under=80
```

CI runs the same check on every push/PR (`.github/workflows/test.yml`).

## Requirements

- **Linux + NVIDIA GPU** for local inference (CUDA 12.4+, ~1× H100/A100-class recommended for int8 avatar)
- **Python 3.10**, [uv](https://github.com/astral-sh/uv)
- Weights downloaded from Hugging Face (~tens of GB)
- For Azure: `az` CLI, subscription with GPU quota, SSH key

macOS can prepare inputs and drive Azure; it cannot run inference locally.

## Quick start — your own assets (Linux GPU)

```bash
# 1. Environment + weights (one-time, ~30–60 min)
bash scripts/setup_uv_env.sh
bash scripts/download_longcat_weights.sh

# 2. Prepare input JSON
LongCat-Video/.venv/bin/python scripts/prepare_longcat_avatar_input.py \
  --image /path/to/portrait.png \
  --audio /path/to/voice.mp3 \
  --prompt "Describe the scene and how the person should speak." \
  --output-json /tmp/my_avatar.json

# 3. Run (int8 + distill by default)
PATH="LongCat-Video/.venv/bin:$PATH" bash scripts/run_longcat_avatar_single.sh /tmp/my_avatar.json
```

Output: `LongCat-Video/outputs_avatar_single/ai2v_demo_1.mp4`

Preflight:

```bash
LongCat-Video/.venv/bin/python scripts/validate_longcat_avatar_setup.py
```

### Azure confidential H100 (NCC) extra step

If you are on `Standard_NCC40ads_H100_v5`, enable the GPU before running:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi conf-compute -srs 1
```

Without this, PyTorch fails with CUDA error 802.

## Quick start — Azure manual VM

```bash
./scripts/launch_longcat_azure_ncc_manual.sh <resource-group> [vm-name] [--delete-on-exit]
```

Example:

```bash
./scripts/launch_longcat_azure_ncc_manual.sh rg-ml-datascience avatar-run-1 --delete-on-exit
```

## Quick start — SkyPilot on Azure

From the control-plane VM (see SPEC):

```bash
source ~/skypilot-runtime/bin/activate
cd ~/avatar-gen
export SKYPILOT_OUTPUT_BUCKET=https://<account>.blob.core.windows.net/longcat-outputs
./scripts/launch_longcat_sky.sh
```

Monitor and tear down:

```bash
sky status
sky logs longcat-ncc-we
sky down longcat-ncc-we -y
```

## Project layout

```
avatar-gen/
  README.md
  SPEC.md
  scripts/                  ← run helpers (this repo)
  skypilot/                 ← Azure SkyPilot YAML
  assets/avatar/custom/     ← sample inputs
  LongCat-Video/            ← submodule (upstream LongCat)
    weights/                ← gitignored; download locally
    outputs_avatar_single/  ← gitignored; generated videos
```

## Troubleshooting

**Submodule empty** — `git submodule update --init --recursive`

**`torchrun: not found`** — `PATH="LongCat-Video/.venv/bin:$PATH"`

**CUDA error 802 on Azure NCC** — `sudo nvidia-smi conf-compute -srs 1`

See [`SPEC.md`](SPEC.md) for SkyPilot patches and Azure quota details.

## License

LongCat-Video upstream is MIT. See [`LongCat-Video/LICENSE`](LongCat-Video/LICENSE).
