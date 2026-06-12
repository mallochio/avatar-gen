#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${SKYPILOT_CLUSTER:-avatar-gen}"
YAML="${REPO_ROOT}/skypilot/avatar.yaml"
NUM_GPUS="${NUM_GPUS:-1}"
USE_SPOT="${USE_SPOT:-1}"

find_input() {
  local base="$1"
  shift
  local ext
  for ext in "$@"; do
    local path="${REPO_ROOT}/inputs/${base}.${ext}"
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

INPUT_IMAGE="$(find_input portrait png jpg jpeg webp || true)"
INPUT_AUDIO="$(find_input audio mp3 wav m4a flac ogg || true)"

if [[ -z "${INPUT_IMAGE}" || -z "${INPUT_AUDIO}" ]]; then
  echo "Place your files in ${REPO_ROOT}/inputs/:" >&2
  echo "  portrait.png  (or .jpg / .jpeg / .webp) — face image" >&2
  echo "  audio.mp3     (or .wav / .m4a / .flac / .ogg) — voice track" >&2
  echo "Optional: prompt.txt — scene description" >&2
  exit 1
fi

case "${NUM_GPUS}" in
  1|2|4|8) ;;
  *)
    echo "NUM_GPUS must be 1, 2, 4, or 8 (got: ${NUM_GPUS})" >&2
    exit 1
    ;;
esac

if ! command -v sky >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/.sky-venv/bin/sky" ]]; then
    export PATH="${REPO_ROOT}/.sky-venv/bin:${PATH}"
  else
    echo "SkyPilot not found. Running setup..." >&2
    bash "${REPO_ROOT}/scripts/setup_skypilot.sh"
    export PATH="${REPO_ROOT}/.sky-venv/bin:${PATH}"
  fi
fi

REL_IMAGE="inputs/$(basename "${INPUT_IMAGE}")"
REL_AUDIO="inputs/$(basename "${INPUT_AUDIO}")"

ENV_ARGS=(
  --env "INPUT_IMAGE=${REL_IMAGE}"
  --env "INPUT_AUDIO=${REL_AUDIO}"
  --env "CONTEXT_PARALLEL_SIZE=${NUM_GPUS}"
  --env "NPROC_PER_NODE=${NUM_GPUS}"
)

SPOT_ARGS=()
if [[ "${USE_SPOT}" == "0" ]]; then
  SPOT_ARGS+=(--no-use-spot)
fi

cd "${REPO_ROOT}"
mkdir -p outputs

echo "Launching avatar generation on GCP (${NUM_GPUS}× H100 spot, europe-west1)..."
echo "  image: ${REL_IMAGE}"
echo "  audio: ${REL_AUDIO}"

sky launch -c "${CLUSTER}" "${YAML}" \
  --gpus "H100:${NUM_GPUS}" \
  "${SPOT_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  --down \
  -y

OUTPUT_FILE="${REPO_ROOT}/outputs/avatar.mp4"
if [[ ! -f "${OUTPUT_FILE}" ]]; then
  echo "Job finished but ${OUTPUT_FILE} was not found." >&2
  echo "Check logs: sky logs ${CLUSTER}" >&2
  exit 1
fi

echo "Done: ${OUTPUT_FILE}"
