#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
VENV="${VENV:-${LONGCAT_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_RUNTIME=0

if [[ "${LONGCAT_FORCE_RUNTIME:-0}" == "1" ]]; then
  INSTALL_RUNTIME=1
elif [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  INSTALL_RUNTIME=1
fi

if [[ ! -d "${LONGCAT_ROOT}" ]]; then
  echo "LongCat submodule missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  uv venv --seed --python "${PYTHON_VERSION}" "${VENV}"
fi

uv pip install --python "${VENV}/bin/python" huggingface_hub

if [[ "${INSTALL_RUNTIME}" != "1" ]]; then
  echo "Created ${VENV}."
  echo "Skipped runtime install: LongCat avatar inference expects Linux + NVIDIA CUDA."
  echo "Set LONGCAT_FORCE_RUNTIME=1 to try the upstream runtime install anyway."
  exit 0
fi

uv pip install --python "${VENV}/bin/python" --index-url "${TORCH_INDEX_URL}" \
  "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0"
uv pip install --python "${VENV}/bin/python" ninja psutil packaging
uv pip install --python "${VENV}/bin/python" --no-build-isolation "flash_attn==2.7.4.post1"
uv pip install --python "${VENV}/bin/python" -r "${LONGCAT_ROOT}/requirements.txt"
grep -Ev '^(libsndfile1|tritonserverclient)' "${LONGCAT_ROOT}/requirements_avatar.txt" \
  | uv pip install --python "${VENV}/bin/python" -r /dev/stdin

echo "Runtime dependencies installed into ${VENV}."
