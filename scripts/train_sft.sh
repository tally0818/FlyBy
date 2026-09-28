#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
set +x
[ -f .env ] && set -a && source .env && set +a
python -m src.train.sft --config configs/sft.yaml "$@"
