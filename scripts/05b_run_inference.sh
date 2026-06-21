#!/usr/bin/env bash
set -euo pipefail

# Inference is deliberately separate from report regeneration. Outputs are
# versioned and existing prediction artifacts are not overwritten by default.
DONUT_CHECKPOINT="${DONUT_CHECKPOINT:-checkpoints/donut/receipt_ie/finetune/best_model}"
LAYOUTXLM_CHECKPOINT="${LAYOUTXLM_CHECKPOINT:-checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

python -m receipt_ie.inference.infer_layoutxlm \
  --checkpoint "$LAYOUTXLM_CHECKPOINT" \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl "outputs/predictions/layoutxlm_test_${RUN_TAG}.jsonl"

python -m receipt_ie.inference.infer_donut \
  --checkpoint "$DONUT_CHECKPOINT" \
  --generation_max_length 768 \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl "outputs/predictions/donut_test_${RUN_TAG}.jsonl"
