#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/eval_cvrr.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Evaluation config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN:-python}" evaluate.py --config "${CONFIG_PATH}" "$@"
