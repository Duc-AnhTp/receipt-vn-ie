#!/usr/bin/env bash
set -euo pipefail

echo "=== Derive evaluation artifacts from frozen predictions ==="

python -m receipt_ie.metrics.summarize_outputs \
  --gold data/processed/test.jsonl \
  --pred \
    outputs/predictions/baseline_test.jsonl \
    outputs/predictions/layoutxlm_test.jsonl \
    outputs/predictions/donut_test.jsonl \
  --metrics_output outputs/metrics/combined_metrics.json \
  --latency_output outputs/metrics/latency_by_method.json

python - <<'PY'
import json
from pathlib import Path

combined = json.loads(
    Path("outputs/metrics/combined_metrics.json").read_text(encoding="utf-8")
)
for method, metrics in combined.items():
    Path(f"outputs/metrics/{method}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
PY

python scripts/build_artifact_manifest.py
python -m receipt_ie.metrics.comparison_analysis
bash scripts/06_error_analysis.sh
python scripts/plot_results.py
python scripts/generate_report_artifacts.py

echo "Derived metrics, analyses, plots and LaTeX tables without inference."
