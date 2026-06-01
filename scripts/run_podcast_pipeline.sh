#!/usr/bin/env bash
# End-to-end two-host podcast: diarize mixed audio → detect bboxes → prepare JSON → infer.
#
# Example (NotebookLM mixed audio + two-person seed image):
#   ./scripts/run_podcast_pipeline.sh \
#     --seed-image /path/to/podcasters.png \
#     --mixed-audio /path/to/notebooklm_overview.mp3 \
#     --prompt "Static camera, two podcast hosts converse naturally in a studio."
#
# If you already have separated tracks, pass --person1-audio and --person2-audio instead of --mixed-audio.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LONGCAT_ROOT="${LONGCAT_ROOT:-${REPO_ROOT}/LongCat-Video}"
VENV="${VENV:-${LONGCAT_ROOT}/.venv}"
PYTHON="${PYTHON:-python3}"
if [[ -x "${VENV}/bin/python" ]]; then
  PYTHON="${VENV}/bin/python"
fi

SEED_IMAGE=""
MIXED_AUDIO=""
PERSON1_AUDIO=""
PERSON2_AUDIO=""
PROMPT=""
WORK_DIR="${WORK_DIR:-/tmp/longcat_podcast}"
OUTPUT_JSON="${OUTPUT_JSON:-${WORK_DIR}/podcast_input.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${LONGCAT_ROOT}/outputs_avatar_multi}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
AUDIO_TYPE="${AUDIO_TYPE:-para}"

usage() {
  sed -n '2,20p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed-image) SEED_IMAGE="$2"; shift 2 ;;
    --mixed-audio) MIXED_AUDIO="$2"; shift 2 ;;
    --person1-audio) PERSON1_AUDIO="$2"; shift 2 ;;
    --person2-audio) PERSON2_AUDIO="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --audio-type) AUDIO_TYPE="$2"; shift 2 ;;
    --prepare-only) RUN_INFERENCE=0; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -n "${SEED_IMAGE}" ]] || { echo "Missing --seed-image" >&2; exit 1; }
[[ -n "${PROMPT}" ]] || { echo "Missing --prompt" >&2; exit 1; }

mkdir -p "${WORK_DIR}"
DIAR_DIR="${WORK_DIR}/diarized"
SPATIAL_JSON="${WORK_DIR}/spatial.json"
DIAR_META="${WORK_DIR}/diarization.json"

if [[ -z "${PERSON1_AUDIO}" || -z "${PERSON2_AUDIO}" ]]; then
  [[ -n "${MIXED_AUDIO}" ]] || {
    echo "Provide --mixed-audio or both --person1-audio and --person2-audio" >&2
    exit 1
  }
  echo "== Step 1: Audio diarization (speaker-separated tracks + silent condition) =="
  "${PYTHON}" "${REPO_ROOT}/scripts/diarize_podcast_audio.py" \
    --mixed-audio "${MIXED_AUDIO}" \
    --output-dir "${DIAR_DIR}" \
    --metadata-json "${DIAR_META}"
  PERSON1_AUDIO="${DIAR_DIR}/person1.wav"
  PERSON2_AUDIO="${DIAR_DIR}/person2.wav"
else
  echo "== Step 1: Using pre-separated speaker tracks (skip diarization) =="
fi

echo "== Step 2: Spatial initialization (bounding boxes + landmarks) =="
"${PYTHON}" "${REPO_ROOT}/scripts/detect_podcast_hosts.py" \
  --image "${SEED_IMAGE}" \
  --output-json "${SPATIAL_JSON}"

# Attach audio duration for multi-clip segment recommendation
"${PYTHON}" - <<PY
import json
from pathlib import Path
import librosa

spatial = json.loads(Path("${SPATIAL_JSON}").read_text(encoding="utf-8"))
y, sr = librosa.load("${PERSON1_AUDIO}", sr=16000, mono=True)
spatial["duration_sec"] = len(y) / float(sr)
Path("${SPATIAL_JSON}").write_text(json.dumps(spatial, indent=2) + "\n", encoding="utf-8")
PY

echo "== Step 3: Multi-person routing input JSON (bbox ↔ audio track mapping) =="
"${PYTHON}" "${REPO_ROOT}/scripts/prepare_longcat_avatar_multi_input.py" \
  --image "${SEED_IMAGE}" \
  --person1-audio "${PERSON1_AUDIO}" \
  --person2-audio "${PERSON2_AUDIO}" \
  --prompt "${PROMPT}" \
  --spatial-json "${SPATIAL_JSON}" \
  --audio-type "${AUDIO_TYPE}" \
  --output-json "${OUTPUT_JSON}"

if [[ "${RUN_INFERENCE}" != "1" ]]; then
  echo "Prepared ${OUTPUT_JSON} (inference skipped)."
  exit 0
fi

echo "== Step 4–6: Image-to-video + long-horizon rollout (Base 50-NFE, Whisper sliding window) =="
echo "    Whisper encoder sliding windows are enabled inside upstream get_audio_embedding_whisper."
echo "    Multi-clip rollout: NUM_SEGMENTS from input JSON (reference latent + continuation)."
if grep -q '"first_frame_hand_presence_check": true' "${OUTPUT_JSON}" 2>/dev/null; then
  echo "    First-frame hand-presence check: enhance_hf enabled via Base model (use_distill=0)."
fi

PATH="${VENV}/bin:${PATH}" OUTPUT_DIR="${OUTPUT_DIR}" \
  bash "${REPO_ROOT}/scripts/run_longcat_avatar_multi.sh" "${OUTPUT_JSON}"

echo "Done. Output directory: ${OUTPUT_DIR}"
