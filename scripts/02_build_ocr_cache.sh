#!/usr/bin/env bash
set -euo pipefail

PREPROCESS_PROFILE="${PREPROCESS_PROFILE:-resize}"
MAX_LONG_SIDE="${MAX_LONG_SIDE:-1600}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "=== Bắt đầu build offline OCR cache (profile: $PREPROCESS_PROFILE, side: $MAX_LONG_SIDE) ==="
python -m receipt_ie.ocr.build_ocr_cache \
  --data_files \
    data/processed/train.jsonl \
    data/processed/val.jsonl \
    data/processed/test.jsonl \
  --preprocess_profile "$PREPROCESS_PROFILE" \
  --max_long_side "$MAX_LONG_SIDE" \
  --batch_size "$BATCH_SIZE" \
  "$@"
