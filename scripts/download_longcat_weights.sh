#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${LONGCAT_ROOT}/weights}"
AVATAR_MODE="int8"
WITH_DISTILL=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp16)
      AVATAR_MODE="fp16"
      ;;
    --no-distill)
      WITH_DISTILL=0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "${WEIGHTS_ROOT}/LongCat-Video" "${WEIGHTS_ROOT}/LongCat-Video-Avatar-1.5"

HF=(uvx --from huggingface_hub hf download)
BASE_ARGS=(
  meituan-longcat/LongCat-Video
  --local-dir "${WEIGHTS_ROOT}/LongCat-Video"
  --include "tokenizer/*"
  --include "text_encoder/*"
  --include "vae/*"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  BASE_ARGS+=(--dry-run)
fi

"${HF[@]}" "${BASE_ARGS[@]}"

AVATAR_ARGS=(
  meituan-longcat/LongCat-Video-Avatar-1.5
  --local-dir "${WEIGHTS_ROOT}/LongCat-Video-Avatar-1.5"
  --include "scheduler/*"
  --include "vocal_separator/*"
  --include "whisper-large-v3/config.json"
  --include "whisper-large-v3/preprocessor_config.json"
  --include "whisper-large-v3/model.safetensors"
  --include "whisper-large-v3/tokenizer.json"
  --include "whisper-large-v3/tokenizer_config.json"
  --include "whisper-large-v3/merges.txt"
  --include "whisper-large-v3/vocab.json"
)

if [[ "${AVATAR_MODE}" == "int8" ]]; then
  AVATAR_ARGS+=(--include "base_model_int8/*")
else
  AVATAR_ARGS+=(--include "base_model/*")
fi

if [[ "${WITH_DISTILL}" == "1" ]]; then
  AVATAR_ARGS+=(--include "lora/dmd_lora.safetensors")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  AVATAR_ARGS+=(--dry-run)
fi

"${HF[@]}" "${AVATAR_ARGS[@]}"

echo "Weights ready under ${WEIGHTS_ROOT}."
