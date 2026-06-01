#!/usr/bin/env bash
# Full podcast pipeline on an Azure NCC H100 VM (manual launch, no SkyPilot).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
VENV="${VENV:-${LONGCAT_ROOT}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${LONGCAT_ROOT}/outputs_avatar_multi}"
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

export USE_DISTILL="${USE_DISTILL:-0}"
export USE_INT8="${USE_INT8:-0}"
export OUTPUT_DIR WORK_DIR OUTPUT_JSON
export PODCAST_SEED_IMAGE="${PODCAST_SEED_IMAGE:-LongCat-Video/assets/avatar/multi/introduce.png}"
export PODCAST_PROMPT="${PODCAST_PROMPT:-Static camera, two podcast hosts converse in a recording studio.}"

if [[ -z "${PODCAST_MIXED_AUDIO:-}" && ( -z "${PODCAST_PERSON1_AUDIO:-}" || -z "${PODCAST_PERSON2_AUDIO:-}" ) ]]; then
  PODCAST_MIXED_AUDIO="/tmp/notebooklm_mixed.wav"
  "${VENV}/bin/python" "${REPO_ROOT}/scripts/make_mixed_podcast_audio.py" \
    --person1-audio LongCat-Video/assets/avatar/multi/introduce_woman.mp3 \
    --person2-audio LongCat-Video/assets/avatar/multi/introduce_man.mp3 \
    --output "${PODCAST_MIXED_AUDIO}"
  export PODCAST_MIXED_AUDIO
fi

bash "${REPO_ROOT}/scripts/run_podcast_pipeline.sh"
ls -lah "${OUTPUT_DIR}"
