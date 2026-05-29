import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from receipt_ie.metrics.evaluate_fields import evaluate_predictions
from receipt_ie.metrics.latency import compute_latency_stats


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(data: Dict[str, Any], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize all prediction files into metrics and latency JSON files.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path")
    parser.add_argument("--pred", nargs="+", required=True, help="Prediction JSONL paths")
    parser.add_argument("--metrics_output", required=True, help="Output combined metrics JSON")
    parser.add_argument("--latency_output", required=True, help="Output latency JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    gold = read_jsonl(args.gold)
    metrics = {}
    latency = {}

    for pred_path in args.pred:
        path = Path(pred_path)
        if not path.exists():
            print(f"Warning: prediction file not found, skipped: {pred_path}")
            continue
        preds = read_jsonl(str(path))
        method = preds[0].get("method", path.stem) if preds else path.stem
        metrics[method] = evaluate_predictions(preds, gold)
        latency[method] = {
            "ocr": compute_latency_stats([float(p.get("latency_ocr_ms", 0.0) or 0.0) for p in preds]),
            "model": compute_latency_stats([float(p.get("latency_model_ms", p.get("latency_cached_ms", 0.0)) or 0.0) for p in preds]),
            "postprocess": compute_latency_stats([float(p.get("latency_postprocess_ms", 0.0) or 0.0) for p in preds]),
            "e2e": compute_latency_stats([float(p.get("latency_e2e_ms", 0.0) or 0.0) for p in preds]),
        }

    write_json(metrics, args.metrics_output)
    write_json(latency, args.latency_output)
    print(f"Saved combined metrics to {args.metrics_output}")
    print(f"Saved latency summary to {args.latency_output}")


if __name__ == "__main__":
    main()
