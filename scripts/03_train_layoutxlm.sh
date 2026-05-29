#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"
LAYOUTXLM_MODE="${LAYOUTXLM_MODE:-ocr_cache}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" -m receipt_ie.training.train_layoutxlm --mode "$LAYOUTXLM_MODE" "$@"
else
  python -m receipt_ie.training.train_layoutxlm --mode "$LAYOUTXLM_MODE" "$@"
fi
