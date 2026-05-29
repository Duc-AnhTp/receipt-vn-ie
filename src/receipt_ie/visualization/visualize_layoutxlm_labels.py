import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from receipt_ie.data.build_layoutxlm_labels import assign_word_labels


COLOR_MAP = {
    "OCR": (59, 130, 246),
    "STORE_NAME": (239, 68, 68),
    "DATE": (34, 197, 94),
    "TOTAL": (249, 115, 22),
    "ADDRESS": (168, 85, 247),
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_ocr_words(sample: Dict[str, Any], ocr_cache_dir: str) -> List[Dict[str, Any]]:
    cache_path = sample.get("ocr_cache_path") or ""
    candidates = []
    if cache_path:
        candidates.append(Path(cache_path))
        candidates.append(Path(ocr_cache_dir) / Path(cache_path).name)
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("words", [])
    return []


def draw_labels(image: Image.Image, words: List[Dict[str, Any]], labels: List[str]) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for word, label in zip(words, labels):
        bbox = word.get("bbox")
        if not bbox:
            continue
        draw.rectangle(bbox, outline=COLOR_MAP["OCR"], width=1)
        if label != "O" and "-" in label:
            field = label.split("-", 1)[1]
            draw.rectangle(bbox, outline=COLOR_MAP.get(field, (107, 114, 128)), width=3)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LayoutXLM BIO labels from OCR cache and field boxes.")
    parser.add_argument("--jsonl", required=True, help="Dataset JSONL path")
    parser.add_argument("--ocr_cache", default="data/interim/ocr_cache", help="OCR cache directory")
    parser.add_argument("--output_dir", default="outputs/debug/bio_labels", help="Output image directory")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of samples")
    parser.add_argument("--project_root", default=".", help="Project root for relative image paths")
    parser.add_argument("--overlap_threshold", type=float, default=0.5, help="Overlap threshold for BIO assignment")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root)

    written = 0
    for sample in read_jsonl(args.jsonl):
        if written >= args.limit:
            break
        if sample.get("annotation_level") != "json_and_boxes":
            continue
        image_path = project_root / sample.get("image_path", "")
        if not image_path.exists():
            continue

        words = load_ocr_words(sample, args.ocr_cache)
        if not words:
            continue

        word_texts = [w.get("text", "") for w in words]
        word_boxes = [w.get("bbox", [0, 0, 0, 0]) for w in words]
        labels = assign_word_labels(
            word_texts,
            word_boxes,
            sample.get("field_boxes", {}),
            overlap_threshold=args.overlap_threshold,
        )
        with Image.open(image_path).convert("RGB") as image:
            drawn = draw_labels(image, words, labels)
        drawn.save(out_dir / f"{sample.get('id', written)}.jpg")
        written += 1

    print(f"Saved {written} BIO label visualizations to {out_dir}")


if __name__ == "__main__":
    main()
