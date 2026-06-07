#!/usr/bin/env bash
# Install and enable NVIDIA GPUs on Azure NCC confidential H100 VMs.
# SkyPilot setup: call once; script reboots when drivers are first installed.
set -euo pipefail

install_ncc_drivers() {
  sudo apt-get update
  local driver_pkg="nvidia-driver-595-server"
  local utils_pkg="nvidia-utils-595-server"
  local kernel module_pkg="" candidate
  local candidates=(
    "linux-modules-nvidia-595-server-$(uname -r)"
    "linux-modules-nvidia-595-server-azure-fde"
    "linux-modules-nvidia-595-server-generic"
  )
  for candidate in "${candidates[@]}"; do
    if apt-cache show "${candidate}" >/dev/null 2>&1; then
      module_pkg="${candidate}"
      break
    fi
  done
  if [[ -z "${module_pkg}" ]]; then
    echo "No NVIDIA kernel module package found. Tried: ${candidates[*]}" >&2
    exit 1
  fi
  echo "Installing ${driver_pkg} with ${module_pkg}"
  sudo apt-get install -y "${driver_pkg}" "${utils_pkg}" "${module_pkg}"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  install_ncc_drivers
  if [[ "${LONGCAT_NCC_SKIP_REBOOT:-0}" != "1" ]]; then
    sudo reboot
    while true; do sleep 5; done
  fi
fi

sudo nvidia-smi -pm 1
sudo nvidia-smi conf-compute -srs 1

mapfile -t gpus < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed '/^$/d')
if [[ "${#gpus[@]}" -lt 1 ]]; then
  echo "NCC GPU setup failed: nvidia-smi reports 0 GPUs" >&2
  nvidia-smi >&2 || true
  exit 1
fi

echo "NCC GPU ready: ${#gpus[@]} device(s)"
nvidia-smi -L
