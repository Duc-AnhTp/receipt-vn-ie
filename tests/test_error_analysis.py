import csv
import json
import unittest
from pathlib import Path

from receipt_ie.metrics.error_analysis import classify_error, main as error_analysis_main
from receipt_ie.metrics.evaluate_fields import evaluate_predictions


class TestErrorAnalysis(unittest.TestCase):
    def test_evaluate_predictions_still_works(self):
        results = evaluate_predictions(
            [{"id": "a", "normalized_prediction": {"store_name": "A", "date": "2026-05-22", "total": "100", "address": "HN"}}],
            [{"id": "a", "target": {"store_name": "A", "date": "2026-05-22", "total": "100", "address": "HN"}}],
        )
        self.assertEqual(results["macro"]["EM"], 1.0)

    def test_error_analysis_cli_writes_csv(self):
        root = Path("tmp/test_error_analysis")
        root.mkdir(parents=True, exist_ok=True)
        gold = root / "gold.jsonl"
        pred = root / "pred.jsonl"
        out = root / "errors.csv"

        gold.write_text(json.dumps({
            "id": "a",
            "target": {"store_name": "A Store", "date": "2026-05-22", "total": "100000", "address": "HN"},
            "ocr_cache_path": "",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        pred.write_text(json.dumps({
            "id": "a",
            "method": "baseline",
            "prediction": {"store_name": "", "date": "", "total": "", "address": ""},
            "normalized_prediction": {"store_name": "", "date": "", "total": "", "address": ""},
            "status": "ok",
        }, ensure_ascii=False) + "\n", encoding="utf-8")

        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "error_analysis",
                "--gold", str(gold),
                "--pred", str(pred),
                "--ocr_cache_dir", str(root),
                "--output", str(out),
            ]
            error_analysis_main()
        finally:
            sys.argv = old_argv

        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        self.assertEqual(len(rows), 4)
        store_row = next(r for r in rows if r["field"] == "store_name")
        self.assertEqual(store_row["error_type"], "EMPTY_PRED")
        summary = root / "errors_summary.csv"
        self.assertTrue(summary.exists())

    def test_donut_never_uses_ocr_error_types(self):
        error_type = classify_error(
            "store_name",
            "A Store",
            "B Store",
            "B Store",
            [{"text": "A St0re"}],
            "donut",
        )
        self.assertEqual(error_type, "MODEL_BAD")

    def test_ocr_method_distinguishes_wrong_and_missing(self):
        wrong = classify_error(
            "store_name",
            "anan",
            "other",
            "other",
            [{"text": "anam"}],
            "layoutxlm",
        )
        missing = classify_error(
            "store_name",
            "anan",
            "other",
            "other",
            [{"text": "completely different"}],
            "baseline",
        )
        self.assertEqual(wrong, "OCR_WRONG")
        self.assertEqual(missing, "OCR_MISS")


if __name__ == "__main__":
    unittest.main()
