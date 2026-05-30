#!/usr/bin/env bash
set -euo pipefail

echo "=== Bắt đầu chạy OCR Preprocess Ablation ==="

for PROFILE in none resize rectify binarize; do
  echo "----------------------------------------"
  echo "Profile: $PROFILE"
  echo "----------------------------------------"
  
  # Chạy build cache cho val set giới hạn 100 mẫu để ablate nhanh
  python -m receipt_ie.ocr.build_ocr_cache \
    --data_files data/processed/val.jsonl \
    --limit 100 \
    --preprocess_profile "$PROFILE" \
    --output_dir "outputs/ocr_ablation/cache_$PROFILE" \
    --overwrite
      
  # Đánh giá chất lượng OCR
  python -m receipt_ie.metrics.ocr_quality \
    --jsonl data/processed/val.jsonl \
    --limit 100 \
    --ocr_cache_dir "outputs/ocr_ablation/cache_$PROFILE" \
    --output_csv "outputs/ocr_ablation/ocr_quality_$PROFILE.csv"
done
