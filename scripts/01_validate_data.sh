#!/usr/bin/env bash
set -euo pipefail

echo "=== Bắt đầu validate dữ liệu Receipt VN IE ==="
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/train.jsonl
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/val.jsonl
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/test.jsonl
