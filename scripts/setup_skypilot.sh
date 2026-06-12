#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.sky-venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  uv venv --seed --python "${PYTHON_VERSION}" "${VENV}"
fi

uv pip install --python "${VENV}/bin/python" --prerelease allow "skypilot[gcp]"

"${VENV}/bin/sky" api stop >/dev/null 2>&1 || true
"${VENV}/bin/sky" check gcp

echo "SkyPilot ready."
echo "Generate: ${REPO_ROOT}/scripts/generate_avatar.sh"
echo "YAML: ${REPO_ROOT}/skypilot/avatar.yaml"
