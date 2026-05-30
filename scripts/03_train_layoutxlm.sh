#!/usr/bin/env bash
set -euo pipefail

USE_ACCELERATE="${USE_ACCELERATE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
LAYOUTXLM_MODE="${LAYOUTXLM_MODE:-ocr_cache}"

echo "=== Bắt đầu huấn luyện LayoutXLM (mode: $LAYOUTXLM_MODE) ==="

if [[ "$USE_ACCELERATE" == "1" ]]; then
  accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" \
    -m receipt_ie.training.train_layoutxlm \
    --mode "$LAYOUTXLM_MODE" "$@"
else
  python -m receipt_ie.training.train_layoutxlm \
    --mode "$LAYOUTXLM_MODE" "$@"
fi
