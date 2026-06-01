#!/usr/bin/env bash
# Launch two-host podcast generation on Azure NCC H100 via SkyPilot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${SKYPILOT_PODCAST_CLUSTER:-longcat-podcast-we}"
YAML="${SKYPILOT_PODCAST_YAML:-${REPO_ROOT}/skypilot/longcat_podcast_azure_westeurope_h100.yaml}"

if [[ -z "${SKYPILOT_OUTPUT_BUCKET:-}" ]]; then
  echo "Set SKYPILOT_OUTPUT_BUCKET to an Azure blob container URL, e.g.:" >&2
  echo "  export SKYPILOT_OUTPUT_BUCKET=https://<account>.blob.core.windows.net/longcat-outputs" >&2
  exit 1
fi

if ! command -v sky >/dev/null 2>&1; then
  echo "SkyPilot CLI not found. Install from https://docs.skypilot.co/en/latest/getting-started/installation.html" >&2
  exit 1
fi

cd "${REPO_ROOT}"

ENV_ARGS=(--env "SKYPILOT_OUTPUT_BUCKET=${SKYPILOT_OUTPUT_BUCKET}")

# Optional podcast inputs (paths relative to repo root are synced in workdir)
[[ -n "${PODCAST_SEED_IMAGE:-}" ]] && ENV_ARGS+=(--env "PODCAST_SEED_IMAGE=${PODCAST_SEED_IMAGE}")
[[ -n "${PODCAST_MIXED_AUDIO:-}" ]] && ENV_ARGS+=(--env "PODCAST_MIXED_AUDIO=${PODCAST_MIXED_AUDIO}")
[[ -n "${PODCAST_PERSON1_AUDIO:-}" ]] && ENV_ARGS+=(--env "PODCAST_PERSON1_AUDIO=${PODCAST_PERSON1_AUDIO}")
[[ -n "${PODCAST_PERSON2_AUDIO:-}" ]] && ENV_ARGS+=(--env "PODCAST_PERSON2_AUDIO=${PODCAST_PERSON2_AUDIO}")
[[ -n "${PODCAST_PROMPT:-}" ]] && ENV_ARGS+=(--env "PODCAST_PROMPT=${PODCAST_PROMPT}")
[[ -n "${PODCAST_AUDIO_TYPE:-}" ]] && ENV_ARGS+=(--env "PODCAST_AUDIO_TYPE=${PODCAST_AUDIO_TYPE}")
[[ -n "${USE_INT8:-}" ]] && ENV_ARGS+=(--env "USE_INT8=${USE_INT8}")
[[ -n "${USE_DISTILL:-}" ]] && ENV_ARGS+=(--env "USE_DISTILL=${USE_DISTILL}")
[[ -n "${NUM_SEGMENTS:-}" ]] && ENV_ARGS+=(--env "NUM_SEGMENTS=${NUM_SEGMENTS}")
[[ -n "${RESOLUTION:-}" ]] && ENV_ARGS+=(--env "RESOLUTION=${RESOLUTION}")


echo "Launching podcast job on cluster: ${CLUSTER}"
echo "YAML: ${YAML}"
echo "Outputs: ${SKYPILOT_OUTPUT_BUCKET}/longcat-podcast-<task-id>/"

sky launch -c "${CLUSTER}" "${YAML}" "${ENV_ARGS[@]}" -y
