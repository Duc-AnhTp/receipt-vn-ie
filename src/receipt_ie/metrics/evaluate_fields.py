from pathlib import Path
from typing import Any, Dict, List

from rapidfuzz.distance import Levenshtein


FIELDS = ["store_name", "date", "total", "address"]


def compute_em(pred: str, gold: str) -> float:
    """Exact match after trimming outer whitespace."""
    return 1.0 if pred.strip() == gold.strip() else 0.0


def compute_nes(pred: str, gold: str) -> float:
    """Normalized edit similarity based on Levenshtein distance."""
    pred = pred.strip()
    gold = gold.strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return 1.0 - Levenshtein.distance(pred, gold) / max(len(pred), len(gold))


def compute_cer(pred: str, gold: str) -> float:
    """Character error rate with an explicit convention for empty gold."""
    pred = pred.strip()
    gold = gold.strip()
    if not gold:
        return 0.0 if not pred else 1.0
    return Levenshtein.distance(pred, gold) / len(gold)


def _empty_metric_lists() -> Dict[str, Dict[str, List[float]]]:
    return {field: {"em": [], "nes": [], "cer": []} for field in FIELDS}


def _append_metrics(bucket: Dict[str, List[float]], pred: str, gold: str) -> None:
    bucket["em"].append(compute_em(pred, gold))
    bucket["nes"].append(compute_nes(pred, gold))
    bucket["cer"].append(compute_cer(pred, gold))


def _summarize(metric_lists: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    macro = {"EM": [], "NES": [], "CER": []}

    for field in FIELDS:
        values = metric_lists[field]
        count = len(values["em"])
        field_result = {
            "EM": round(sum(values["em"]) / count, 4) if count else 0.0,
            "NES": round(sum(values["nes"]) / count, 4) if count else 0.0,
            "CER": round(sum(values["cer"]) / count, 4) if count else 0.0,
            "n_samples": count,
        }
        results[field] = field_result
        for metric in macro:
            macro[metric].append(field_result[metric])

    results["macro"] = {
        metric: round(sum(values) / len(values), 4)
        for metric, values in macro.items()
    }
    results["macro"]["averaging"] = "unweighted_mean_over_fields"
    return results


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate predictions in two views:

    - top-level field metrics: all test samples, including empty ground truth;
    - ``present_only``: only samples whose ground truth for that field is non-empty.

    Inference errors and missing prediction IDs are evaluated as empty predictions.
    This prevents system failures from being silently excluded from the score.
    """
    all_metrics = _empty_metric_lists()
    present_metrics = _empty_metric_lists()
    coverage_counts = {
        field: {"nonempty_predictions": 0, "total_samples": 0}
        for field in FIELDS
    }

    has_ids = any(gold.get("id") for gold in ground_truths)
    prediction_map = {
        prediction.get("id"): prediction
        for prediction in predictions
        if prediction.get("id")
    } if has_ids else {}

    n_inference_errors = 0
    missing_prediction_ids: List[str] = []
    n_evaluated = 0

    if has_ids:
        pairs = []
        for gold in ground_truths:
            gold_id = gold.get("id")
            if not gold_id:
                continue
            prediction = prediction_map.get(gold_id)
            if prediction is None:
                missing_prediction_ids.append(gold_id)
            pairs.append((prediction, gold))
    else:
        pairs = [
            (predictions[index] if index < len(predictions) else None, gold)
            for index, gold in enumerate(ground_truths)
        ]

    for prediction, gold in pairs:
        prediction_failed = prediction is None or prediction.get("status") == "error"
        if prediction is not None and prediction.get("status") == "error":
            n_inference_errors += 1

        pred_data = {} if prediction_failed else (
            prediction.get("normalized_prediction") or prediction
        )
        gold_data = gold.get("target") or gold

        for field in FIELDS:
            pred_value = str(pred_data.get(field, "") or "")
            gold_value = str(gold_data.get(field, "") or "")
            _append_metrics(all_metrics[field], pred_value, gold_value)

            coverage_counts[field]["total_samples"] += 1
            if pred_value.strip():
                coverage_counts[field]["nonempty_predictions"] += 1

            if gold_value.strip():
                _append_metrics(present_metrics[field], pred_value, gold_value)

        n_evaluated += 1

    results = _summarize(all_metrics)
    results["present_only"] = _summarize(present_metrics)
    results["prediction_coverage"] = {
        field: {
            **counts,
            "rate": round(
                counts["nonempty_predictions"] / counts["total_samples"], 4
            ) if counts["total_samples"] else 0.0,
        }
        for field, counts in coverage_counts.items()
    }
    results["n_ground_truth_present"] = {
        field: results["present_only"][field]["n_samples"]
        for field in FIELDS
    }
    results["n_evaluated"] = n_evaluated
    results["n_inference_errors"] = n_inference_errors
    results["missing_prediction_ids"] = missing_prediction_ids
    # Legacy key retained for downstream consumers. Errors are no longer skipped.
    results["n_skipped_error"] = 0
    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Đánh giá dự đoán trường thông tin hóa đơn.")
    parser.add_argument("--gold", "--gold_jsonl", required=True)
    parser.add_argument("--pred", "--pred_jsonl", required=True)
    parser.add_argument(
        "--output",
        "--output_json",
        default="outputs/metrics/main_metrics.json",
    )
    args = parser.parse_args()

    def read_jsonl(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    results = evaluate_predictions(read_jsonl(args.pred), read_jsonl(args.gold))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f"Saved evaluation metrics to {output_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
