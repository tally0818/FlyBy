#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec python -m src.eval.main "$@"
