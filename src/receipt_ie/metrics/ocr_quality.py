import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from rapidfuzz import fuzz

from receipt_ie.data.schemas import FIELDS


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def norm(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_ocr_text(sample: Dict[str, Any], ocr_cache_dir: str) -> str:
    cache_path = sample.get("ocr_cache_path") or ""
    candidates = []
    if cache_path:
        candidates.append(Path(cache_path))
        candidates.append(Path(ocr_cache_dir) / Path(cache_path).name)
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return " ".join(w.get("text", "") for w in data.get("words", []) if w.get("text"))
    return ""


def rough_contains(gold: str, ocr_text: str) -> bool:
    gold_n = norm(gold)
    ocr_n = norm(ocr_text)
    if not gold_n:
        return True
    if gold_n in ocr_n:
        return True
    return fuzz.partial_ratio(gold_n, ocr_n) >= 80


def parse_args():
    parser = argparse.ArgumentParser(description="Create a rough OCR quality sample report.")
    parser.add_argument("--jsonl_path", required=True, help="Dataset JSONL path")
    parser.add_argument("--ocr_cache_dir", default="data/interim/ocr_cache", help="OCR cache directory")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=30, help="Maximum number of samples")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = read_jsonl(args.jsonl_path)[: args.limit]
    rows = []
    for sample in samples:
        ocr_text = load_ocr_text(sample, args.ocr_cache_dir)
        target = sample.get("target", {})
        for field in FIELDS:
            gold = target.get(field, "")
            rows.append({
                "id": sample.get("id", ""),
                "field": field,
                "gold_text": gold,
                "ocr_text_joined": ocr_text,
                "ocr_contains_gold_rough": rough_contains(gold, ocr_text),
                "note": "",
            })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "field", "gold_text", "ocr_text_joined", "ocr_contains_gold_rough", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved OCR quality sample with {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
