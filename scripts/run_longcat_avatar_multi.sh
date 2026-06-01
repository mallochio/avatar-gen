#!/usr/bin/env bash
# Multi-person LongCat-Video-Avatar-1.5 inference (podcast / dual-host).
# Defaults: Base model 50-NFE (no distill), FP16 DiT, multi-clip rollout via num_segments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
VENV="${VENV:-${LONGCAT_ROOT}/.venv}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.json>" >&2
  echo "Build JSON: scripts/prepare_longcat_avatar_multi_input.py or scripts/run_podcast_pipeline.sh" >&2
  exit 1
fi

INPUT_JSON="$1"
OUTPUT_DIR="${OUTPUT_DIR:-${LONGCAT_ROOT}/outputs_avatar_multi}"
MODEL_TYPE="${MODEL_TYPE:-avatar-v1.5}"
RESOLUTION="${RESOLUTION:-480p}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LONGCAT_ROOT}/weights/LongCat-Video-Avatar-1.5}"

# Step 5: Base model (50-NFE) — disable distill / 8-NFE accelerated path
USE_DISTILL="${USE_DISTILL:-0}"
USE_INT8="${USE_INT8:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TEXT_GUIDANCE_SCALE="${TEXT_GUIDANCE_SCALE:-4.0}"
AUDIO_GUIDANCE_SCALE="${AUDIO_GUIDANCE_SCALE:-4.0}"

REF_IMG_INDEX="${REF_IMG_INDEX:-10}"
MASK_FRAME_RANGE="${MASK_FRAME_RANGE:-3}"

if [[ -z "${NUM_SEGMENTS:-}" ]] && command -v python3 >/dev/null 2>&1; then
  NUM_SEGMENTS="$(python3 - "${INPUT_JSON}" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
print(data.get("recommended_num_segments", data.get("num_segments", 1)))
PY
)"
fi
NUM_SEGMENTS="${NUM_SEGMENTS:-1}"

TORCHRUN="${TORCHRUN:-torchrun}"
if [[ -x "${VENV}/bin/torchrun" ]]; then
  TORCHRUN="${VENV}/bin/torchrun"
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "${TORCHRUN}"
  --nproc_per_node="${NPROC_PER_NODE}"
  "${LONGCAT_ROOT}/run_demo_avatar_multi_audio_to_video.py"
  --context_parallel_size="${CONTEXT_PARALLEL_SIZE}"
  --checkpoint_dir="${CHECKPOINT_DIR}"
  --input_json="${INPUT_JSON}"
  --output_dir="${OUTPUT_DIR}"
  --resolution="${RESOLUTION}"
  --num_segments="${NUM_SEGMENTS}"
  --num_inference_steps="${NUM_INFERENCE_STEPS}"
  --text_guidance_scale="${TEXT_GUIDANCE_SCALE}"
  --audio_guidance_scale="${AUDIO_GUIDANCE_SCALE}"
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

printf 'Running multi-person avatar (Base 50-NFE=%s, segments=%s):\n' \
  "$( [[ "${USE_DISTILL}" == "0" ]] && echo yes || echo no )" "${NUM_SEGMENTS}"
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
