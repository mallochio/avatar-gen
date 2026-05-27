#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <resource-group> [vm-name] [--delete-on-exit]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCE_GROUP="$1"
VM_NAME="${2:-longcat-ncc-manual}"
DELETE_ON_EXIT=0
if [[ "${3:-}" == "--delete-on-exit" ]] || [[ "${2:-}" == "--delete-on-exit" ]]; then
  DELETE_ON_EXIT=1
  if [[ "${2:-}" == "--delete-on-exit" ]]; then
    VM_NAME="longcat-ncc-manual"
  fi
fi

LOCATION="${AZURE_LOCATION:-westeurope}"
VM_SIZE="${AZURE_NCC_VM_SIZE:-Standard_NCC40ads_H100_v5}"
IDENTITY="${AZURE_MANAGED_IDENTITY:-mi-aml-intern}"
OUTPUT_DIR="${REPO_ROOT}/outputs_avatar_single"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519.pub}"
REMOTE_DIR="avatar-gen"

if [[ ! -f "${SSH_KEY}" ]]; then
  SSH_KEY="${HOME}/.ssh/id_rsa.pub"
fi
if [[ ! -f "${SSH_KEY}" ]]; then
  echo "No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub" >&2
  exit 1
fi

cleanup() {
  if [[ "${DELETE_ON_EXIT}" == "1" ]]; then
    echo "Deleting VM ${VM_NAME}..."
    az vm delete -g "${RESOURCE_GROUP}" -n "${VM_NAME}" --yes >/dev/null 2>&1 || true
  fi
}
if [[ "${DELETE_ON_EXIT}" == "1" ]]; then
  trap cleanup EXIT
fi

echo "Creating confidential H100 VM ${VM_NAME} in ${LOCATION}..."
az vm create \
  -g "${RESOURCE_GROUP}" \
  -n "${VM_NAME}" \
  --location "${LOCATION}" \
  --size "${VM_SIZE}" \
  --image Canonical:0001-com-ubuntu-confidential-vm-jammy:22_04-lts-cvm:latest \
  --admin-user azureuser \
  --assign-identity "${IDENTITY}" \
  --security-type ConfidentialVM \
  --enable-secure-boot true \
  --enable-vtpm true \
  --os-disk-security-encryption-type DiskWithVMGuestState \
  --public-ip-sku Standard \
  --ssh-key-values "@${SSH_KEY}" \
  --os-disk-size-gb 512 \
  --output none

PUBLIC_IP="$(az vm show -d -g "${RESOURCE_GROUP}" -n "${VM_NAME}" --query publicIps -o tsv)"
echo "VM public IP: ${PUBLIC_IP}"

echo "Waiting for SSH..."
for _ in $(seq 1 30); do
  if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "azureuser@${PUBLIC_IP}" true 2>/dev/null; then
    break
  fi
  sleep 10
done

echo "Syncing repo..."
rsync -avz \
  --exclude weights --exclude outputs_avatar_single --exclude .venv --exclude .sky-venv \
  --exclude '.git' --exclude 'LongCat-Video/.git' --exclude 'LongCat-Video/weights' \
  --exclude 'LongCat-Video/.venv' --exclude 'LongCat-Video/outputs_avatar_single' \
  "${REPO_ROOT}/" "azureuser@${PUBLIC_IP}:~/${REMOTE_DIR}/"

echo "Running setup and inference (this takes a while)..."
ssh -o StrictHostKeyChecking=no "azureuser@${PUBLIC_IP}" \
  "bash ~/${REMOTE_DIR}/scripts/setup_and_run_longcat_ncc.sh"

mkdir -p "${OUTPUT_DIR}"
scp -o StrictHostKeyChecking=no \
  "azureuser@${PUBLIC_IP}:~/${REMOTE_DIR}/LongCat-Video/outputs_avatar_single/*" \
  "${OUTPUT_DIR}/"

echo "Output saved under ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
