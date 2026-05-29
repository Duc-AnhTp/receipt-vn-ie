#!/usr/bin/env bash
set -euo pipefail

python -m receipt_ie.ocr.build_ocr_cache "$@"
python -m receipt_ie.metrics.ocr_quality \
  --jsonl_path data/processed/train.jsonl \
  --ocr_cache_dir data/interim/ocr_cache \
  --output_csv outputs/ocr/ocr_quality_sample.csv \
  --limit 30
