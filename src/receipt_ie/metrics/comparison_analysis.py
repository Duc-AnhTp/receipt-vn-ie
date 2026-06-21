"""Paired comparison and sensitivity analyses from frozen prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from receipt_ie.metrics.evaluate_fields import (
    FIELDS,
    compute_em,
    compute_nes,
    evaluate_predictions,
)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prediction_map(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(item["id"]): item for item in records if item.get("id")}


def _value(record: Dict[str, Any] | None, field: str) -> str:
    if record is None or record.get("status") == "error":
        return ""
    data = record.get("normalized_prediction") or record
    return str(data.get(field, "") or "")


def paired_bootstrap_present_only(
    ground_truths: List[Dict[str, Any]],
    baseline_predictions: List[Dict[str, Any]],
    layoutxlm_predictions: List[Dict[str, Any]],
    *,
    seed: int = 20260620,
    n_resamples: int = 50_000,
    batch_size: int = 2_000,
) -> Dict[str, Any]:
    """Bootstrap LayoutXLM minus Baseline using present-only field means."""
    baseline_map = _prediction_map(baseline_predictions)
    layoutxlm_map = _prediction_map(layoutxlm_predictions)
    n_samples = len(ground_truths)

    masks: Dict[str, np.ndarray] = {}
    differences: Dict[str, Dict[str, np.ndarray]] = {
        "EM": {},
        "NES": {},
    }
    for field in FIELDS:
        present = []
        em_diff = []
        nes_diff = []
        for gold in ground_truths:
            sample_id = str(gold.get("id", ""))
            target = gold.get("target") or gold
            gold_value = str(target.get(field, "") or "")
            present.append(bool(gold_value.strip()))
            baseline_value = _value(baseline_map.get(sample_id), field)
            layoutxlm_value = _value(layoutxlm_map.get(sample_id), field)
            em_diff.append(
                compute_em(layoutxlm_value, gold_value)
                - compute_em(baseline_value, gold_value)
            )
            nes_diff.append(
                compute_nes(layoutxlm_value, gold_value)
                - compute_nes(baseline_value, gold_value)
            )
        masks[field] = np.asarray(present, dtype=np.float64)
        differences["EM"][field] = np.asarray(em_diff, dtype=np.float64)
        differences["NES"][field] = np.asarray(nes_diff, dtype=np.float64)

    observed = {}
    for metric in ("EM", "NES"):
        field_means = [
            float(
                np.sum(differences[metric][field] * masks[field])
                / np.sum(masks[field])
            )
            for field in FIELDS
        ]
        observed[metric] = float(np.mean(field_means))

    rng = np.random.default_rng(seed)
    sampled_differences = {
        metric: np.empty(n_resamples, dtype=np.float64)
        for metric in ("EM", "NES")
    }
    offset = 0
    while offset < n_resamples:
        current = min(batch_size, n_resamples - offset)
        indices = rng.integers(0, n_samples, size=(current, n_samples))
        for metric in ("EM", "NES"):
            field_bootstrap_means = []
            for field in FIELDS:
                sampled_mask = masks[field][indices]
                denominator = sampled_mask.sum(axis=1)
                numerator = (
                    differences[metric][field][indices] * sampled_mask
                ).sum(axis=1)
                field_bootstrap_means.append(
                    np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(numerator),
                        where=denominator > 0,
                    )
                )
            sampled_differences[metric][offset:offset + current] = np.mean(
                np.vstack(field_bootstrap_means),
                axis=0,
            )
        offset += current

    result: Dict[str, Any] = {
        "comparison": "layoutxlm_minus_baseline",
        "view": "present_only",
        "seed": seed,
        "n_resamples": n_resamples,
        "n_samples": n_samples,
        "interpretation": (
            "Sampling uncertainty on the frozen test set only; this does not "
            "remove training-data or configuration bias."
        ),
        "metrics": {},
    }
    for metric in ("EM", "NES"):
        values = sampled_differences[metric]
        result["metrics"][metric] = {
            "observed_difference": round(observed[metric], 6),
            "ci95_percentile": [
                round(float(np.quantile(values, 0.025)), 6),
                round(float(np.quantile(values, 0.975)), 6),
            ],
            "probability_positive": round(float(np.mean(values > 0)), 6),
        }
    return result


def sensitivity_analysis(
    ground_truths: List[Dict[str, Any]],
    predictions_by_method: Dict[str, List[Dict[str, Any]]],
    excluded_ids: Iterable[str],
) -> Dict[str, Any]:
    excluded = set(excluded_ids)
    filtered_gold = [
        item for item in ground_truths if str(item.get("id", "")) not in excluded
    ]
    filtered_ids = {str(item.get("id", "")) for item in filtered_gold}
    metrics = {}
    for method, predictions in predictions_by_method.items():
        filtered_predictions = [
            item
            for item in predictions
            if str(item.get("id", "")) in filtered_ids
        ]
        metrics[method] = evaluate_predictions(filtered_predictions, filtered_gold)
    return {
        "excluded_ids": sorted(excluded),
        "n_samples": len(filtered_gold),
        "reason": "raw_image_and_ocr_cache_coordinate_size_mismatch",
        "metrics": metrics,
    }


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired bootstrap and coordinate sensitivity analyses."
    )
    parser.add_argument("--gold", default="data/processed/test.jsonl")
    parser.add_argument(
        "--baseline",
        default="outputs/predictions/baseline_test.jsonl",
    )
    parser.add_argument(
        "--layoutxlm",
        default="outputs/predictions/layoutxlm_test.jsonl",
    )
    parser.add_argument(
        "--bootstrap_output",
        default="outputs/metrics/paired_bootstrap.json",
    )
    parser.add_argument(
        "--sensitivity_output",
        default="outputs/metrics/sensitivity_230.json",
    )
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--n_resamples", type=int, default=50_000)
    parser.add_argument(
        "--exclude_ids",
        nargs="+",
        default=[
            "self_img_272",
            "self_img_340",
            "self_img_458",
            "self_img_188",
        ],
    )
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    baseline = read_jsonl(args.baseline)
    layoutxlm = read_jsonl(args.layoutxlm)
    write_json(
        paired_bootstrap_present_only(
            gold,
            baseline,
            layoutxlm,
            seed=args.seed,
            n_resamples=args.n_resamples,
        ),
        args.bootstrap_output,
    )
    write_json(
        sensitivity_analysis(
            gold,
            {"baseline": baseline, "layoutxlm": layoutxlm},
            args.exclude_ids,
        ),
        args.sensitivity_output,
    )


if __name__ == "__main__":
    main()
