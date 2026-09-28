#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

exec python -m src.search_r1.retrieval_server \
  --index-path "${SEARCH_R1_INDEX_PATH:-data/search_r1/wiki18/e5_Flat.index}" \
  --corpus-path "${SEARCH_R1_CORPUS_PATH:-data/search_r1/wiki18/wiki-18.jsonl}" \
  --model "${SEARCH_R1_RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
  --model-revision "${SEARCH_R1_RETRIEVER_REVISION:-f52bf8ec8c7124536f0efb74aca902b2995e5bcd}" \
  --host "${SEARCH_R1_RETRIEVER_HOST:-127.0.0.1}" \
  --port "${SEARCH_R1_RETRIEVER_PORT:-8000}" \
  --topk "${SEARCH_R1_TOPK:-3}" \
  "$@"
