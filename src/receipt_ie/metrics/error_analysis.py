import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rapidfuzz import fuzz

from receipt_ie.data.schemas import FIELDS
from receipt_ie.metrics.evaluate_fields import compute_cer, compute_em, compute_nes


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def norm_for_match(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def joined_ocr_text(sample: Dict[str, Any], ocr_cache_dir: str) -> str:
    cache_path = sample.get("ocr_cache_path") or ""
    candidates = []
    if cache_path:
        candidates.append(Path(cache_path))
        candidates.append(Path(ocr_cache_dir) / Path(cache_path).name)

    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                words = [w.get("text", "") for w in data.get("words", []) if w.get("text")]
                if words:
                    return " ".join(words)
                lines = []
                for line in data.get("lines", []):
                    lines.append(" ".join(w.get("text", "") for w in line if w.get("text")))
                return " ".join(lines)
            except Exception:
                return ""
    return ""


def has_format_error(field: str, value: str) -> bool:
    if not value:
        return False
    if field == "date":
        return DATE_RE.match(value) is None
    if field == "total":
        return not value.isdigit()
    return False


def classify_error(
    method: str,
    field: str,
    gold: str,
    pred: str,
    raw_pred: str,
    ocr_text: str,
    em: float,
    nes: float,
) -> str:
    if em == 1.0:
        return "OK"

    norm_gold = norm_for_match(gold)
    norm_ocr = norm_for_match(ocr_text)
    ocr_rough_score = fuzz.partial_ratio(norm_gold, norm_ocr) if norm_gold and norm_ocr else 0
    ocr_has_gold = bool(norm_gold) and (norm_gold in norm_ocr or ocr_rough_score >= 80)

    if not pred:
        if method == "layoutxlm" and ocr_has_gold:
            return "LABEL_BAD"
        return "EMPTY_PRED"
    if has_format_error(field, pred):
        return "FORMAT_ERROR"

    if method == "donut":
        if raw_pred and pred != raw_pred:
            raw_nes = compute_nes(str(raw_pred), gold)
            if raw_nes > nes + 0.15:
                return "POSTPROCESS_BAD"
        return "MODEL_BAD"

    if norm_gold and norm_gold not in norm_ocr:
        return "OCR_WRONG" if ocr_rough_score >= 60 else "OCR_MISS"

    if raw_pred and pred != raw_pred:
        raw_nes = compute_nes(str(raw_pred), gold)
        if raw_nes > nes + 0.15:
            return "POSTPROCESS_BAD"

    if method == "layoutxlm" and ocr_has_gold:
        return "LABEL_BAD"
    return "MODEL_BAD"


def prediction_rows(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    ocr_cache_dir: str,
) -> Iterable[Dict[str, Any]]:
    gold_by_id = {r.get("id"): r for r in gold_records if r.get("id")}
    pred_by_id = {r.get("id"): r for r in pred_records if r.get("id")}

    for sample_id, gold_record in gold_by_id.items():
        pred_record = pred_by_id.get(sample_id, {})
        method = pred_record.get("method", "unknown")
        pred_data = pred_record.get("normalized_prediction") or {}
        raw_data = pred_record.get("prediction") or {}
        gold_data = gold_record.get("target") or {}
        ocr_text = joined_ocr_text(gold_record, ocr_cache_dir)

        for field in FIELDS:
            gold = str(gold_data.get(field) or "")
            pred = str(pred_data.get(field) or "")
            raw_pred = str(raw_data.get(field) or "")
            em = compute_em(pred, gold)
            nes = compute_nes(pred, gold)
            cer = compute_cer(pred, gold)
            yield {
                "id": sample_id,
                "method": method,
                "field": field,
                "gold": gold,
                "pred": raw_pred,
                "normalized_pred": pred,
                "em": em,
                "nes": round(nes, 4),
                "cer": round(cer, 4),
                "error_type": classify_error(method, field, gold, pred, raw_pred, ocr_text, em, nes),
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Create field-level error analysis CSV.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path")
    parser.add_argument("--pred", nargs="+", required=True, help="One or more prediction JSONL paths")
    parser.add_argument("--ocr_cache_dir", default="data/interim/ocr_cache", help="OCR cache directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    gold_records = read_jsonl(args.gold)
    rows = []
    for pred_path in args.pred:
        if not Path(pred_path).exists():
            print(f"Warning: prediction file not found, skipped: {pred_path}")
            continue
        rows.extend(prediction_rows(gold_records, read_jsonl(pred_path), args.ocr_cache_dir))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "method", "field", "gold", "pred", "normalized_pred", "em", "nes", "cer", "error_type"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved error analysis with {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
