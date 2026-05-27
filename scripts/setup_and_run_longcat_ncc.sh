#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
VENV="${VENV:-${LONGCAT_ROOT}/.venv}"
INPUT_JSON="${1:-/tmp/longcat_input.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LONGCAT_ROOT}/outputs_avatar_single}"
TEST_IMAGE="${TEST_IMAGE:-${REPO_ROOT}/assets/avatar/custom/random_person.png}"
TEST_AUDIO="${TEST_AUDIO:-${REPO_ROOT}/assets/avatar/custom/random_voice.mp3}"
TEST_PROMPT="${TEST_PROMPT:-A photorealistic office worker delivers a dramatic monologue with total sincerity, as if revealing a ridiculous secret while trying very hard to stay professional.}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

export PATH="${HOME}/.local/bin:${CUDA_HOME}/bin:${VENV}/bin:${PATH}"

sudo nvidia-smi -pm 1 || true
sudo nvidia-smi conf-compute -srs 1 || true

if ! command -v nvcc >/dev/null 2>&1; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg git build-essential libsndfile1 libsndfile1-dev
  sudo apt-get install -y nvidia-utils-595-server || true
  export LONGCAT_FORCE_RUNTIME=1
  bash "${REPO_ROOT}/scripts/setup_uv_env.sh"
  bash "${REPO_ROOT}/scripts/download_longcat_weights.sh"
fi

"${VENV}/bin/python" "${REPO_ROOT}/scripts/prepare_longcat_avatar_input.py" \
  --image "${TEST_IMAGE}" \
  --audio "${TEST_AUDIO}" \
  --prompt "${TEST_PROMPT}" \
  --output-json "${INPUT_JSON}"

OUTPUT_DIR="${OUTPUT_DIR}" bash "${REPO_ROOT}/scripts/run_longcat_avatar_single.sh" "${INPUT_JSON}"
ls -lah "${OUTPUT_DIR}"
