#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SKYPILOT_API_SERVER_ENDPOINT="${SKYPILOT_API_SERVER_ENDPOINT:-http://127.0.0.1:46580}"
CLUSTER="${SKYPILOT_CLUSTER:-avatar-gen}"
YAML="${REPO_ROOT}/skypilot/avatar.yaml"
NUM_GPUS="${NUM_GPUS:-1}"
USE_SPOT="${USE_SPOT:-1}"
AVATAR_OUTPUT_BUCKET="${AVATAR_OUTPUT_BUCKET:-}"

sky() {
  uv run --group cloud sky "$@"
}

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

if [[ -z "${AVATAR_OUTPUT_BUCKET}" ]]; then
  echo "Set AVATAR_OUTPUT_BUCKET to your GCS bucket, e.g.:" >&2
  echo "  export AVATAR_OUTPUT_BUCKET=gs://your-avatar-output-bucket" >&2
  exit 1
fi

AVATAR_OUTPUT_BUCKET="${AVATAR_OUTPUT_BUCKET%/}"
export AVATAR_OUTPUT_BUCKET

case "${NUM_GPUS}" in
  1|2|4|8) ;;
  *)
    echo "NUM_GPUS must be 1, 2, 4, or 8 (got: ${NUM_GPUS})" >&2
    exit 1
    ;;
esac

cd "${REPO_ROOT}"
if ! uv run --group cloud sky --version >/dev/null 2>&1; then
  echo "SkyPilot not found. Running setup..." >&2
  bash "${REPO_ROOT}/scripts/setup_skypilot.sh"
fi

uv run --group cloud sky api stop >/dev/null 2>&1 || true

REL_IMAGE="inputs/$(basename "${INPUT_IMAGE}")"
REL_AUDIO="inputs/$(basename "${INPUT_AUDIO}")"

ENV_ARGS=(
  --env "INPUT_IMAGE=${REL_IMAGE}"
  --env "INPUT_AUDIO=${REL_AUDIO}"
  --env "CONTEXT_PARALLEL_SIZE=${NUM_GPUS}"
  --env "NPROC_PER_NODE=${NUM_GPUS}"
)

if [[ -n "${NUM_SEGMENTS:-}" ]]; then
  ENV_ARGS+=(--env "NUM_SEGMENTS=${NUM_SEGMENTS}")
fi

SPOT_ARGS=()
if [[ "${USE_SPOT}" == "0" ]]; then
  SPOT_ARGS+=(--no-use-spot)
fi

mkdir -p outputs

echo "Launching avatar generation on GCP (${NUM_GPUS}× H100 spot, europe-west1)..."
echo "  image: ${REL_IMAGE}"
echo "  audio: ${REL_AUDIO}"
echo "  output: ${AVATAR_OUTPUT_BUCKET}/avatar-gen/latest/avatar.mp4"

sky launch -c "${CLUSTER}" "${YAML}" \
  --gpus "H100:${NUM_GPUS}" \
  "${SPOT_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  --down \
  -y

gcloud storage cp "${AVATAR_OUTPUT_BUCKET}/avatar-gen/latest/avatar.mp4" outputs/avatar.mp4

OUTPUT_FILE="${REPO_ROOT}/outputs/avatar.mp4"
if [[ ! -f "${OUTPUT_FILE}" ]]; then
  echo "Job finished but ${OUTPUT_FILE} was not found." >&2
  echo "Check logs: uv run --group cloud sky logs ${CLUSTER}" >&2
  exit 1
fi

echo "Done: ${OUTPUT_FILE}"
