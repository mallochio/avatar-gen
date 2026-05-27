#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
VENV="${VENV:-${LONGCAT_ROOT}/.venv}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.json>" >&2
  echo "Generate one with: scripts/prepare_longcat_avatar_input.py --image ... --audio ... --prompt ... --output-json /tmp/input.json" >&2
  exit 1
fi
INPUT_JSON="$1"
OUTPUT_DIR="${OUTPUT_DIR:-${LONGCAT_ROOT}/outputs_avatar_single}"
MODEL_TYPE="${MODEL_TYPE:-avatar-v1.5}"
RESOLUTION="${RESOLUTION:-480p}"
NUM_SEGMENTS="${NUM_SEGMENTS:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
USE_INT8="${USE_INT8:-1}"
USE_DISTILL="${USE_DISTILL:-1}"
REF_IMG_INDEX="${REF_IMG_INDEX:-10}"
MASK_FRAME_RANGE="${MASK_FRAME_RANGE:-3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LONGCAT_ROOT}/weights/LongCat-Video-Avatar-1.5}"

TORCHRUN="${TORCHRUN:-torchrun}"
if [[ -x "${VENV}/bin/torchrun" ]]; then
  TORCHRUN="${VENV}/bin/torchrun"
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${TORCHRUN}"
  --nproc_per_node="${NPROC_PER_NODE}"
  "${LONGCAT_ROOT}/run_demo_avatar_single_audio_to_video.py"
  --context_parallel_size="${CONTEXT_PARALLEL_SIZE}"
  --checkpoint_dir="${CHECKPOINT_DIR}"
  --stage_1=ai2v
  --input_json="${INPUT_JSON}"
  --output_dir="${OUTPUT_DIR}"
  --resolution="${RESOLUTION}"
  --num_segments="${NUM_SEGMENTS}"
  --ref_img_index="${REF_IMG_INDEX}"
  --mask_frame_range="${MASK_FRAME_RANGE}"
  --model_type="${MODEL_TYPE}"
)

if [[ "${USE_DISTILL}" == "1" ]]; then
  CMD+=(--use_distill)
fi
if [[ "${USE_INT8}" == "1" ]]; then
  CMD+=(--use_int8)
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
