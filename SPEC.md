# LongCat Avatar — Azure / SkyPilot Spec

## Goal

Run **LongCat-Video-Avatar-1.5** for single-image + matching-audio avatar generation on Azure GPU, with optional SkyPilot orchestration and blob output.

Primary task: `ai2v` with `avatar-v1.5`, `--use_distill`, `--use_int8`.

## Current verdict (May 2026)

**Inference works.** A full run produced `outputs_avatar_single/ai2v_demo_1.mp4` from `random_person.png` + `random_voice.mp3`.

**Azure GPU path that works in this subscription:**

- SKU: `Standard_NCC40ads_H100_v5` (confidential H100)
- Region: `westeurope`
- Quota family: `Standard NCCads2023 Family vCPUs` (160 vCPU limit; 40 vCPU per VM)

**Azure GPU paths that do not work here:**

- T4 / A10 / V100 / non-confidential H100 in US regions — zero quota
- SkyPilot default T4 YAMLs — never provisioned in this subscription

## What succeeded

### Manual Azure VM (proven end-to-end)

1. `az vm create` with ConfidentialVM + `Standard_NCC40ads_H100_v5`
2. GPU driver install + `sudo nvidia-smi conf-compute -srs 1` (required for CUDA on NCC)
3. Repo setup via `scripts/setup_and_run_longcat_ncc.sh`
4. Output: `~/outputs_avatar_single/ai2v_demo_1.mp4`

Wrapper: `scripts/launch_longcat_azure_ncc_manual.sh`

### SkyPilot interactive cluster (provision verified)

After VM-side patches (ConfidentialVM image, `SubResource` fix, `mi-aml-intern` identity, westeurope catalog entry):

- `sky launch -c longcat-ncc-we skypilot/longcat_avatar_azure_westeurope_h100.yaml` **provisioned** an NCC H100 in westeurope
- Setup (CUDA, flash_attn, HF weights) takes 20–40+ minutes; full run was interrupted during teardown tests

Canonical YAML: `skypilot/longcat_avatar_azure_westeurope_h100.yaml`

### Local Linux + NVIDIA (supported by scripts)

- `scripts/setup_uv_env.sh` — creates `.venv` on Linux with CUDA
- `scripts/run_longcat_avatar_single.sh` — single-GPU inference entrypoint
- macOS: validate weights/assets only; no local inference (needs Linux + CUDA + NCCL)

## Control plane (Azure VM)

SkyPilot API server runs on dedicated Azure VM (not laptop):

- VM: `sky-api-235928` @ `51.124.153.185:46580`
- SSH: `azureuser@51.124.153.185`
- Run SkyPilot from VM: `source ~/skypilot-runtime/bin/activate`

Required `~/.sky/config.yaml` on control-plane VM:

```yaml
jobs:
  bucket: https://<storage-account>.blob.core.windows.net/longcat-outputs

azure:
  resource_group_vm: rg-ml-datascience
  storage_account: <storage-account>
  remote_identity: mi-aml-intern
```

Mac client: NSG must allow your IP on port `46580`. Set `api_server.endpoint` in local `~/.sky/config.yaml` if driving from laptop.

**VM-side SkyPilot patches** (re-apply after `pip install -U skypilot`):

1. `network.SubResource` instead of `compute.SubResource` in `sky/provision/azure/instance.py` (lines ~209, ~226)
2. ConfidentialVM block in `_create_vm` for `NCC` instance types (image + security profile)
3. Append `Standard_NCC40ads_H100_v5` row for `westeurope` to `~/.sky/catalogs/v8/azure/vms.csv` if missing

Fetch catalog: `python -m sky.catalog.data_fetchers.fetch_azure --regions westeurope`

## Azure prerequisites

- Subscription with **NCCads2023** quota in `westeurope` (check: `az vm list-usage --location westeurope`)
- Resource group + storage account + `longcat-outputs` blob container
- User-assigned identity `mi-aml-intern` (or equivalent) with:
  - Contributor on resource group (VM create)
  - **Storage Blob Data Contributor** on storage account (for SkyPilot `/outputs` mount)
- SkyPilot `remote_identity: mi-aml-intern` avoids needing `Microsoft.Authorization/roleAssignments/write`

## Important files

### Run path (local / manual VM)

- `scripts/prepare_longcat_avatar_input.py` — build input JSON from image + audio + prompt
- `scripts/download_longcat_weights.sh` — HF weights into `LongCat-Video/weights/`
- `scripts/setup_uv_env.sh` — Linux CUDA venv
- `scripts/setup_and_run_longcat_ncc.sh` — full setup + run on GPU VM
- `scripts/run_longcat_avatar_single.sh` — `torchrun` wrapper
- `scripts/validate_longcat_avatar_setup.py` — preflight checks

### Azure launch

- `scripts/launch_longcat_azure_ncc_manual.sh` — provision VM, sync repo, run, fetch output
- `scripts/launch_longcat_sky.sh` — SkyPilot launch helper
- `skypilot/longcat_avatar_azure_westeurope_h100.yaml` — SkyPilot task spec

### Test assets

- `assets/avatar/custom/random_person.png`
- `assets/avatar/custom/random_voice.mp3`

Scripts live at repo root; upstream code is in `LongCat-Video/` submodule.

## Run commands (quick reference)

### Your own assets on a Linux GPU machine

```bash
cd avatar-gen
bash scripts/setup_uv_env.sh          # once
bash scripts/download_longcat_weights.sh

LongCat-Video/.venv/bin/python scripts/prepare_longcat_avatar_input.py \
  --image /path/to/face.png \
  --audio /path/to/voice.mp3 \
  --prompt "Your scene description." \
  --output-json /tmp/input.json

PATH="LongCat-Video/.venv/bin:$PATH" bash scripts/run_longcat_avatar_single.sh /tmp/input.json
# output: LongCat-Video/outputs_avatar_single/ai2v_demo_1.mp4
```

On **NCC confidential H100**, before inference:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi conf-compute -srs 1
```

### Azure manual (from laptop with `az` CLI)

```bash
cd avatar-gen
./scripts/launch_longcat_azure_ncc_manual.sh rg-ml-datascience my-run --delete-on-exit
```

### SkyPilot (from control-plane VM)

```bash
ssh azureuser@51.124.153.185
source ~/skypilot-runtime/bin/activate
cd ~/avatar-gen
./scripts/launch_longcat_sky.sh
sky logs longcat-ncc-we    # monitor
sky down longcat-ncc-we -y # teardown
```

## Known constraints

- **NCC H100 is confidential compute**: must use ConfidentialVM + `conf-compute -srs 1`; standard NC H100 SKUs have zero quota here
- **flash_attn**: build with `--no-build-isolation` and CUDA toolkit (`cuda-nvcc-12-4`)
- **requirements_avatar.txt**: skip unpublishable pins `libsndfile1`, `tritonserverclient`; install `libsndfile1-dev` via apt
- **Blob mount**: fails without Storage Blob Data role on the VM identity; fallback is `sky scp` or manual launch script SCP
- **T4/A10/V100 SkyPilot YAMLs**: removed; not viable in this subscription

## Blocker history (resolved vs open)

| Issue | Status |
|-------|--------|
| Mac SkyPilot BrokenProcessPool | Bypassed: API server on Azure VM |
| Missing python/sky on job controller | Fixed: runtime shim on VM |
| T4/A10/V100 quota | Blocked: zero quota; use NCC only |
| westeurope not in SkyPilot catalog | Fixed: manual catalog row + fetch |
| roleAssignments/write RBAC | Fixed: `remote_identity: mi-aml-intern` |
| SubResource JSON error | Fixed: VM patch |
| ConfidentialVM security type | Fixed: VM patch + manual az path |
| CUDA error 802 on NCC | Fixed: `nvidia-smi conf-compute -srs 1` |
| SkyPilot full setup+run to blob | Partial: provision OK; validate end-to-end after blob IAM |

## Teardown

```bash
sky down longcat-ncc-we -y                    # SkyPilot cluster
az vm delete -g rg-ml-datascience -n <name> --yes  # manual VM
# If delete blocked, use REST:
az rest --method DELETE --url "/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Compute/virtualMachines/<name>?api-version=2024-11-01"
```

Keep `sky-api-235928` running if you still use remote SkyPilot.
