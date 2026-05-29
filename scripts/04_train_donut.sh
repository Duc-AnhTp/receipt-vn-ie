#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"
DONUT_MODE="${DONUT_MODE:-finetune}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" -m receipt_ie.training.train_donut --mode "$DONUT_MODE" "$@"
else
  python -m receipt_ie.training.train_donut --mode "$DONUT_MODE" "$@"
fi
