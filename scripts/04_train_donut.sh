#!/usr/bin/env bash
set -euo pipefail

USE_ACCELERATE="${USE_ACCELERATE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

echo "=== Bắt đầu huấn luyện Donut ==="

if [[ "$USE_ACCELERATE" == "1" ]]; then
  accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" \
    -m receipt_ie.training.train_donut \
    --mode finetune "$@"
else
  python -m receipt_ie.training.train_donut \
    --mode finetune "$@"
fi
