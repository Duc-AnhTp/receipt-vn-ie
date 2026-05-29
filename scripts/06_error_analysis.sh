#!/usr/bin/env bash
set -euo pipefail

python -m receipt_ie.metrics.error_analysis \
  --gold data/processed/test.jsonl \
  --pred outputs/predictions/baseline_test.jsonl outputs/predictions/layoutxlm_test.jsonl outputs/predictions/donut_test.jsonl \
  --ocr_cache_dir data/interim/ocr_cache \
  --output outputs/error_analysis/error_by_field.csv
