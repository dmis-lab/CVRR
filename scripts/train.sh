#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/train_cvrr_qwen25_7b.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

# Keep device assignment under the caller's control.  When CUDA_VISIBLE_DEVICES
# is set, infer one worker per visible device unless NPROC_PER_NODE overrides it.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _cvrr_devices <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_cvrr_devices[@]}"
  else
    NPROC_PER_NODE=1
  fi
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  exec "${TORCHRUN_BIN:-torchrun}" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py --config "${CONFIG_PATH}" "$@"
fi

exec "${PYTHON_BIN:-python}" train.py --config "${CONFIG_PATH}" "$@"
