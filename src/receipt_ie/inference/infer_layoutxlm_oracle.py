"""Inference LayoutXLM sử dụng oracle OCR (ground-truth text + boxes)."""
import torch
import json
import time
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from receipt_ie.inference.infer_layoutxlm import LayoutXLMExtractor
from receipt_ie.inference.artifact_metadata import write_inference_sidecar


def main():
    parser = argparse.ArgumentParser(
        description="Chạy suy luận LayoutXLM Oracle trên tập test."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/layoutxlm/receipt_ie/oracle_ocr/best_model",
    )
    parser.add_argument("--test_jsonl", default="data/processed/test.jsonl")
    parser.add_argument(
        "--output_jsonl",
        default="outputs/predictions/layoutxlm_oracle_test.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    extractor = LayoutXLMExtractor()
    extractor.load(args.checkpoint)

    samples = [
        json.loads(line)
        for line in open(args.test_jsonl, encoding="utf-8")
        if line.strip()
    ]
    if args.limit:
        samples = samples[: args.limit]

    oracle_count = sum(1 for s in samples if s.get("oracle_ocr"))
    print(f"Oracle OCR: {oracle_count}/{len(samples)} mẫu có dữ liệu")

    out_path = Path(args.output_jsonl)
    if out_path.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. "
            "Choose a new path or pass --allow_overwrite."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="Oracle Inference"):
            record = {
                "id": sample["id"],
                "method": "layoutxlm_oracle",
                "prediction": {},
                "normalized_prediction": {},
                "latency_ocr_ms": 0.0,
                "latency_model_ms": 0.0,
                "latency_postprocess_ms": 0.0,
                "latency_e2e_ms": 0.0,
                "status": "ok",
                "error": None,
            }
            try:
                oracle_data = sample.get("oracle_ocr", [])
                if not oracle_data:
                    # Mẫu không có oracle → trả rỗng
                    record["prediction"] = {
                        "store_name": "",
                        "date": "",
                        "total": "",
                        "address": "",
                    }
                    record["normalized_prediction"] = record["prediction"].copy()
                else:
                    img_path = sample.get("image_path", "")
                    if not os.path.exists(img_path):
                        raise FileNotFoundError(f"Image not found: {img_path}")

                    img = Image.open(img_path).convert("RGB")
                    words = [w["text"] for w in oracle_data]
                    boxes = [w["box"] for w in oracle_data]

                    res = extractor.predict_from_ocr(img, words, boxes)
                    record["prediction"] = res["prediction"]
                    record["normalized_prediction"] = res["normalized_prediction"]
                    record["latency_model_ms"] = round(res["latency_model_ms"], 2)
                    record["latency_e2e_ms"] = round(res["latency_model_ms"], 2)
            except Exception as e:
                print(f"Error on {sample.get('id')}: {e}")
                record["status"] = "error"
                record["error"] = str(e)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} predictions to {args.output_jsonl}")
    sidecar = write_inference_sidecar(
        out_path,
        method="layoutxlm_oracle",
        checkpoint=args.checkpoint,
        device=extractor.device,
        prediction_count=count,
        inference_arguments={
            "ocr_source": "oracle_ground_truth",
            "n_oracle_samples": oracle_count,
        },
    )
    print(f"Inference metadata saved to {sidecar}")


if __name__ == "__main__":
    main()
