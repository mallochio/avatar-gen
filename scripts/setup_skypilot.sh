#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export SKYPILOT_API_SERVER_ENDPOINT="${SKYPILOT_API_SERVER_ENDPOINT:-http://127.0.0.1:46580}"

uv sync --group cloud --prerelease allow

uv run --group cloud sky api stop >/dev/null 2>&1 || true
uv run --group cloud sky check gcp

echo "SkyPilot ready."
echo "Generate: ${REPO_ROOT}/scripts/generate_avatar.sh"
echo "YAML: ${REPO_ROOT}/skypilot/avatar.yaml"
