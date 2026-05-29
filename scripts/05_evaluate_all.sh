#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/predictions outputs/metrics

TEST_JSONL="${TEST_JSONL:-data/processed/test.jsonl}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_LAYOUTXLM="${RUN_LAYOUTXLM:-1}"
RUN_DONUT="${RUN_DONUT:-1}"
LAYOUTXLM_CHECKPOINT="${LAYOUTXLM_CHECKPOINT:-checkpoints/layoutxlm/receipt_ie/final}"
DONUT_CHECKPOINT="${DONUT_CHECKPOINT:-checkpoints/donut/receipt_ie/final}"

LIMIT_ARG=()
if [ "${LIMIT:-}" != "" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

PRED_FILES=()

if [ "$RUN_BASELINE" = "1" ]; then
  python -m receipt_ie.inference.infer_baseline \
    --test_jsonl "$TEST_JSONL" \
    --output_jsonl outputs/predictions/baseline_test.jsonl \
    "${LIMIT_ARG[@]}"
  PRED_FILES+=(outputs/predictions/baseline_test.jsonl)
else
  echo "Skipping Baseline because RUN_BASELINE=$RUN_BASELINE"
fi

if [ "$RUN_LAYOUTXLM" = "1" ]; then
  if [ -d "$LAYOUTXLM_CHECKPOINT" ]; then
    python -m receipt_ie.inference.infer_layoutxlm \
      --checkpoint "$LAYOUTXLM_CHECKPOINT" \
      --test_jsonl "$TEST_JSONL" \
      --output_jsonl outputs/predictions/layoutxlm_test.jsonl \
      --use_ocr_cache \
      "${LIMIT_ARG[@]}"
    PRED_FILES+=(outputs/predictions/layoutxlm_test.jsonl)
  else
    echo "Skipping LayoutXLM because checkpoint not found: $LAYOUTXLM_CHECKPOINT"
  fi
else
  echo "Skipping LayoutXLM because RUN_LAYOUTXLM=$RUN_LAYOUTXLM"
fi

if [ "$RUN_DONUT" = "1" ]; then
  if [ -d "$DONUT_CHECKPOINT" ]; then
    python -m receipt_ie.inference.infer_donut \
      --checkpoint "$DONUT_CHECKPOINT" \
      --test_jsonl "$TEST_JSONL" \
      --output_jsonl outputs/predictions/donut_test.jsonl \
      "${LIMIT_ARG[@]}"
    PRED_FILES+=(outputs/predictions/donut_test.jsonl)
  else
    echo "Skipping Donut because checkpoint not found: $DONUT_CHECKPOINT"
  fi
else
  echo "Skipping Donut because RUN_DONUT=$RUN_DONUT"
fi

if [ -f outputs/predictions/baseline_test.jsonl ]; then
  python -m receipt_ie.metrics.evaluate_fields \
    --gold "$TEST_JSONL" \
    --pred outputs/predictions/baseline_test.jsonl \
    --output outputs/metrics/baseline_metrics.json
fi
if [ -f outputs/predictions/layoutxlm_test.jsonl ]; then
  python -m receipt_ie.metrics.evaluate_fields \
    --gold "$TEST_JSONL" \
    --pred outputs/predictions/layoutxlm_test.jsonl \
    --output outputs/metrics/layoutxlm_metrics.json
fi
if [ -f outputs/predictions/donut_test.jsonl ]; then
  python -m receipt_ie.metrics.evaluate_fields \
    --gold "$TEST_JSONL" \
    --pred outputs/predictions/donut_test.jsonl \
    --output outputs/metrics/donut_metrics.json
fi

if [ "${#PRED_FILES[@]}" -gt 0 ]; then
  python -m receipt_ie.metrics.summarize_outputs \
    --gold "$TEST_JSONL" \
    --pred "${PRED_FILES[@]}" \
    --metrics_output outputs/metrics/main_metrics.json \
    --latency_output outputs/metrics/latency_by_method.json
else
  echo "No prediction files were generated. Nothing to summarize."
fi
