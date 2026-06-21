import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rapidfuzz.distance import Levenshtein

from receipt_ie.postprocess.total_extractor import strip_accents


ERROR_TYPES = [
    "NONE",
    "EMPTY_PRED",
    "FORMAT_ERROR",
    "OCR_MISS",
    "OCR_WRONG",
    "POSTPROCESS_BAD",
    "MODEL_BAD",
    "GT_EMPTY_PRED_NONEMPTY",
]
OCR_BASED_METHODS = {"baseline", "layoutxlm"}


def normalize_for_compare(value: str) -> str:
    value = strip_accents(value or "").lower()
    return re.sub(r"[^\w\s]", "", value).strip()


def _ocr_similarity(gold_clean: str, ocr_words: List[Dict[str, Any]]) -> float:
    """Best normalized similarity against OCR word windows."""
    tokens = [
        normalize_for_compare(word.get("text", ""))
        for word in ocr_words
        if normalize_for_compare(word.get("text", ""))
    ]
    if not tokens or not gold_clean:
        return 0.0

    gold_token_count = max(1, len(gold_clean.split()))
    candidates = tokens[:]
    for window_size in range(max(1, gold_token_count - 1), gold_token_count + 2):
        candidates.extend(
            " ".join(tokens[index:index + window_size])
            for index in range(0, max(0, len(tokens) - window_size + 1))
        )
    return max(
        Levenshtein.normalized_similarity(gold_clean, candidate)
        for candidate in candidates
    )


def classify_error(
    field: str,
    gold_val: str,
    pred_val: str,
    raw_pred_val: str,
    ocr_words: List[Dict[str, Any]],
    method: str,
) -> str:
    """Assign one deterministic error type to a field prediction."""
    gold_val = str(gold_val or "").strip()
    pred_val = str(pred_val or "").strip()
    raw_pred_val = str(raw_pred_val or "").strip()
    method = method.lower()

    if gold_val == pred_val:
        return "NONE"
    if gold_val and not pred_val:
        return "EMPTY_PRED"
    if not gold_val and pred_val:
        # This observation alone cannot establish whether the label or the
        # prediction is wrong, so use a neutral, reproducible category.
        return "GT_EMPTY_PRED_NONEMPTY"

    if field == "date" and pred_val and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", pred_val):
        return "FORMAT_ERROR"
    if field == "total" and pred_val and not pred_val.isdigit():
        return "FORMAT_ERROR"

    gold_clean = normalize_for_compare(gold_val)
    raw_clean = normalize_for_compare(raw_pred_val)
    pred_clean = normalize_for_compare(pred_val)
    if raw_clean and raw_clean != pred_clean and (
        gold_clean in raw_clean
        or Levenshtein.normalized_similarity(gold_clean, raw_clean) > 0.8
    ):
        return "POSTPROCESS_BAD"

    if method in OCR_BASED_METHODS:
        ocr_text_clean = normalize_for_compare(
            " ".join(word.get("text", "") for word in ocr_words)
        )
        if gold_clean and gold_clean not in ocr_text_clean:
            if _ocr_similarity(gold_clean, ocr_words) >= 0.7:
                return "OCR_WRONG"
            return "OCR_MISS"

    return "MODEL_BAD"


def _read_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("id"):
                    records[item["id"]] = item
    return records


def _load_ocr_words(
    gold_item: Dict[str, Any],
    sample_id: str,
    ocr_cache_dir: Path,
) -> List[Dict[str, Any]]:
    candidates = []
    cache_path = gold_item.get("ocr_cache_path")
    if cache_path:
        candidates.extend([Path(cache_path), ocr_cache_dir / Path(cache_path).name])
    candidates.append(ocr_cache_dir / f"{sample_id}.json")

    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle).get("words", [])
            except (OSError, json.JSONDecodeError):
                continue
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích lỗi có thể tái lập.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--method", choices=["baseline", "layoutxlm", "donut"])
    parser.add_argument("--ocr_cache_dir", default="data/interim/ocr_cache")
    parser.add_argument("--output", default="outputs/error_analysis/error_by_field.csv")
    parser.add_argument("--summary_output", default=None)
    args = parser.parse_args()

    gold_records = _read_jsonl(args.gold)
    pred_records = _read_jsonl(args.pred)
    inferred_method = next(
        (item.get("method") for item in pred_records.values() if item.get("method")),
        Path(args.pred).stem.split("_")[0],
    )
    method = (args.method or inferred_method).lower()

    fields = ["store_name", "date", "total", "address"]
    error_counts = {
        field: {error_type: 0 for error_type in ERROR_TYPES}
        for field in fields
    }
    error_records = []
    ocr_cache_dir = Path(args.ocr_cache_dir)

    for sample_id, gold_item in gold_records.items():
        pred_item = pred_records.get(sample_id) or {}
        gold_target = gold_item.get("target") or gold_item
        prediction_failed = pred_item.get("status") == "error"
        pred_normalized = {} if prediction_failed else (
            pred_item.get("normalized_prediction") or pred_item
        )
        pred_raw = {} if prediction_failed else (pred_item.get("prediction") or {})
        ocr_words = (
            _load_ocr_words(gold_item, sample_id, ocr_cache_dir)
            if method in OCR_BASED_METHODS
            else []
        )

        for field in fields:
            gold_value = gold_target.get(field, "")
            pred_value = pred_normalized.get(field, "")
            raw_value = pred_raw.get(field, "")
            error_type = classify_error(
                field,
                gold_value,
                pred_value,
                raw_value,
                ocr_words,
                method,
            )
            error_counts[field][error_type] += 1
            if error_type != "NONE":
                error_records.append({
                    "id": sample_id,
                    "method": method,
                    "field": field,
                    "gold": gold_value,
                    "pred": pred_value,
                    "raw_pred": raw_value,
                    "error_type": error_type,
                })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        error_records,
        columns=["id", "method", "field", "gold", "pred", "raw_pred", "error_type"],
    ).to_csv(output_path, index=False, encoding="utf-8")

    summary_rows = []
    for field in fields:
        row = {"method": method, "field": field}
        row.update(error_counts[field])
        summary_rows.append(row)
    summary_path = Path(args.summary_output) if args.summary_output else (
        output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Detailed errors saved to: {output_path}")
    print(f"Error summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
