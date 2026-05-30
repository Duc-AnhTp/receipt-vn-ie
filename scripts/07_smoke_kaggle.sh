#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

python -m pytest -p no:cacheprovider \
  tests/test_normalize_text.py \
  tests/test_total_extractor.py \
  tests/test_address_extractor.py \
  tests/test_image_preprocess.py \
  tests/test_ocr_cache_schema.py \
  tests/test_donut_dataset.py \
  tests/test_layoutxlm_labels.py \
  tests/test_layoutxlm_spans.py \
  -q

bash scripts/01_validate_data.sh

python -m receipt_ie.ocr.build_ocr_cache \
  --data_files data/processed/train.jsonl \
  --limit 5 \
  --preprocess_profile "${PREPROCESS_PROFILE:-resize}" \
  --overwrite
