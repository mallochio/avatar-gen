#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${SKYPILOT_CLUSTER:-longcat-ncc-we}"
YAML="${SKYPILOT_YAML:-${REPO_ROOT}/skypilot/longcat_avatar_azure_westeurope_h100.yaml}"

if [[ -z "${SKYPILOT_OUTPUT_BUCKET:-}" ]]; then
  echo "Set SKYPILOT_OUTPUT_BUCKET to an Azure blob container URL, e.g.:" >&2
  echo "  export SKYPILOT_OUTPUT_BUCKET=https://<account>.blob.core.windows.net/longcat-outputs" >&2
  exit 1
fi

cd "${REPO_ROOT}"
sky launch -c "${CLUSTER}" "${YAML}" \
  --env "SKYPILOT_OUTPUT_BUCKET=${SKYPILOT_OUTPUT_BUCKET}" \
  -y
