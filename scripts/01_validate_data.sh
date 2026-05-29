#!/usr/bin/env bash
set -euo pipefail

python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/train.jsonl --report_dir outputs/metrics
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/val.jsonl --report_dir outputs/metrics
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/test.jsonl --report_dir outputs/metrics
