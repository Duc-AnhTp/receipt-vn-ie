#!/usr/bin/env bash
set -euo pipefail

python -m receipt_ie.training.train_layoutxlm --mode "${LAYOUTXLM_MODE:-ocr_cache}" "$@"
