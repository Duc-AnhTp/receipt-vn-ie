#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/predictions outputs/metrics

TEST_JSONL="${TEST_JSONL:-data/processed/test.jsonl}"
LIMIT_ARG=()
if [ "${LIMIT:-}" != "" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

python -m receipt_ie.inference.infer_baseline \
  --test_jsonl "$TEST_JSONL" \
  --output_jsonl outputs/predictions/baseline_test.jsonl \
  "${LIMIT_ARG[@]}"
python -m receipt_ie.inference.infer_layoutxlm \
  --checkpoint "${LAYOUTXLM_CHECKPOINT:-checkpoints/layoutxlm/receipt_ie/final}" \
  --test_jsonl "$TEST_JSONL" \
  --output_jsonl outputs/predictions/layoutxlm_test.jsonl \
  --use_ocr_cache \
  "${LIMIT_ARG[@]}"
python -m receipt_ie.inference.infer_donut \
  --checkpoint "${DONUT_CHECKPOINT:-checkpoints/donut/receipt_ie/final}" \
  --test_jsonl "$TEST_JSONL" \
  --output_jsonl outputs/predictions/donut_test.jsonl \
  "${LIMIT_ARG[@]}"

python -m receipt_ie.metrics.evaluate_fields \
  --gold "$TEST_JSONL" \
  --pred outputs/predictions/baseline_test.jsonl \
  --output outputs/metrics/baseline_metrics.json
python -m receipt_ie.metrics.evaluate_fields \
  --gold "$TEST_JSONL" \
  --pred outputs/predictions/layoutxlm_test.jsonl \
  --output outputs/metrics/layoutxlm_metrics.json
python -m receipt_ie.metrics.evaluate_fields \
  --gold "$TEST_JSONL" \
  --pred outputs/predictions/donut_test.jsonl \
  --output outputs/metrics/donut_metrics.json

python -m receipt_ie.metrics.summarize_outputs \
  --gold "$TEST_JSONL" \
  --pred outputs/predictions/baseline_test.jsonl outputs/predictions/layoutxlm_test.jsonl outputs/predictions/donut_test.jsonl \
  --metrics_output outputs/metrics/main_metrics.json \
  --latency_output outputs/metrics/latency_by_method.json
