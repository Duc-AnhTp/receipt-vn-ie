#!/usr/bin/env bash
set -euo pipefail

DONUT_CHECKPOINT="${DONUT_CHECKPOINT:-checkpoints/donut/receipt_ie/finetune/best_model}"
LAYOUTXLM_CHECKPOINT="${LAYOUTXLM_CHECKPOINT:-checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model}"

echo "=== Bắt đầu chạy Inference cho các mô hình ==="

# 1. Chạy Baseline Heuristics
echo "Running Baseline..."
python -m receipt_ie.inference.infer_baseline \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/baseline_test.jsonl
  
# 2. Chạy LayoutXLM
echo "Running LayoutXLM..."
python -m receipt_ie.inference.infer_layoutxlm \
  --checkpoint "$LAYOUTXLM_CHECKPOINT" \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/layoutxlm_test.jsonl
  
# 3. Chạy Donut
echo "Running Donut..."
python -m receipt_ie.inference.infer_donut \
  --checkpoint "$DONUT_CHECKPOINT" \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/donut_test.jsonl
  
echo "=== Đánh giá kết quả (metrics) ==="
python -m receipt_ie.metrics.evaluate_fields \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/baseline_test.jsonl \
  --output outputs/metrics/baseline_metrics.json
  
python -m receipt_ie.metrics.evaluate_fields \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/layoutxlm_test.jsonl \
  --output outputs/metrics/layoutxlm_metrics.json
  
python -m receipt_ie.metrics.evaluate_fields \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/donut_test.jsonl \
  --output outputs/metrics/donut_metrics.json
  
echo "Inference and evaluation completed."
