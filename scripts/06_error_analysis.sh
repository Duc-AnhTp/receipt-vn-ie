#!/usr/bin/env bash
set -euo pipefail

echo "=== Chạy phân tích lỗi mô hình ==="

# Baseline Heuristics error analysis
python -m receipt_ie.metrics.error_analysis \
  --method baseline \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/baseline_test.jsonl \
  --ocr_cache_dir data/interim/ocr_cache \
  --output outputs/error_analysis/baseline_error_by_field.csv \
  --summary_output outputs/error_analysis/baseline_error_summary.csv
  
# LayoutXLM error analysis
python -m receipt_ie.metrics.error_analysis \
  --method layoutxlm \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/layoutxlm_test.jsonl \
  --ocr_cache_dir data/interim/ocr_cache \
  --output outputs/error_analysis/layoutxlm_error_by_field.csv \
  --summary_output outputs/error_analysis/layoutxlm_error_summary.csv
  
# Donut error analysis
python -m receipt_ie.metrics.error_analysis \
  --method donut \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/donut_test.jsonl \
  --ocr_cache_dir data/interim/ocr_cache \
  --output outputs/error_analysis/donut_error_by_field.csv \
  --summary_output outputs/error_analysis/donut_error_summary.csv
  
echo "Error analysis completed."
