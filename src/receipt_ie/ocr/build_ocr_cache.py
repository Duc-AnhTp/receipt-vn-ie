"""Offline PaddleOCR detection + VietOCR recognition cache builder."""
import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from tqdm import tqdm
from PIL import Image

from receipt_ie.ocr.detect_paddle import crop_region, detect_text_regions, load_paddle_detector
from receipt_ie.ocr.reading_order import sort_reading_order
from receipt_ie.ocr.recognize_vietocr import load_vietocr_model, recognize_regions
from receipt_ie.preprocessing.image_preprocess import (
    PreprocessResult,
    load_receipt_image,
    preprocess_receipt_image,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline OCR cache.")
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=["data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/test.jsonl"],
        help="Dataset JSONL files to OCR-cache.",
    )
    parser.add_argument("--config_ocr", type=str, default="configs/ocr.yaml", help="OCR config path.")
    parser.add_argument(
        "--preprocess_profile",
        type=str,
        default="resize",
        choices=["none", "resize", "rectify", "binarize", "ocr_best"],
        help="Image preprocessing profile before OCR.",
    )
    parser.add_argument("--max_long_side", type=int, default=1600, help="Resize long side limit.")
    parser.add_argument("--cache_version", type=str, default="v2", help="OCR cache schema/version marker.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional cache output dir. If omitted, each sample's ocr_cache_path is used.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per input JSONL for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for VietOCR recognizer.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path_for_sample(sample: Dict[str, Any], output_dir: Path | None) -> Path | None:
    if output_dir is not None:
        return output_dir / f"{_safe_stem(str(sample.get('id') or Path(sample.get('image_path', '')).stem))}.json"
    cache_path = sample.get("ocr_cache_path")
    return Path(cache_path) if cache_path else None


def save_preprocessed_image(pre: PreprocessResult, sample_id: str, output_dir: Path) -> Path:
    profile_dir = output_dir / pre.profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{_safe_stem(sample_id)}_{pre.profile}.png"
    pre.image.save(path)
    return path



def build_cache_base(
    sample_id: str,
    image_path: Path,
    image_hash: str,
    ocr_engine: Dict[str, Any],
    pre: PreprocessResult,
    cache_version: str,
    preprocessed_image_path: Path,
) -> Dict[str, Any]:
    width, height = pre.image.size
    preprocess_meta = {
        "cache_version": cache_version,
        "profile": pre.profile,
        "max_long_side": pre.metadata.get("max_long_side"),
        "scale_x": pre.scale_x,
        "scale_y": pre.scale_y,
        "metadata": pre.metadata,
    }
    return {
        "id": sample_id,
        "image_path": image_path.as_posix(),
        "image_hash": image_hash,
        "ocr_engine": ocr_engine,
        "preprocess": preprocess_meta,
        "preprocess_version": cache_version,
        "preprocess_applied": bool(pre.metadata.get("steps")),
        "coordinate_transform": pre.metadata.get("coordinate_transform", "identity"),
        "image_size": [width, height],
        "original_size": [pre.metadata["original_width"], pre.metadata["original_height"]],
        "preprocessed_size": [width, height],
        "preprocessed_image_path": preprocessed_image_path.as_posix(),
    }


def make_ocr_engine_metadata(det_config: dict, rec_config: dict, rec_engine_config: str) -> Dict[str, Any]:
    return {
        "detector": "PaddleOCR",
        "recognizer": "VietOCR",
        "det_config": {
            "lang": det_config.get("lang", "vi"),
            "use_angle_cls": det_config.get("use_angle_cls", True),
            "gpu": det_config.get("gpu", True),
        },
        "rec_config": rec_engine_config,
        "recognizer_gpu": rec_config.get("gpu", True),
    }


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    args = parse_args()
    if not os.path.exists(args.config_ocr):
        logger.error("Config file not found: %s", args.config_ocr)
        return

    config = load_config(args.config_ocr)
    det_config = config.get("detection", {})
    rec_config = config.get("recognition", {})
    cache_config = config.get("cache", {})

    has_cuda = torch.cuda.is_available()
    use_gpu = det_config.get("gpu", True) and has_cuda
    rec_gpu = rec_config.get("gpu", True) and has_cuda
    rec_engine_config = rec_config.get("default_config", "vgg_transformer")
    y_threshold = cache_config.get("reading_order_y_threshold", 12)

    default_cache_dir = Path(cache_config.get("dir", "data/interim/ocr_cache"))
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_dir = Path(
        cache_config.get("preprocessed_image_dir", "data/interim/preprocessed_images")
        if output_dir is None
        else output_dir / "_preprocessed_images"
    )

    logger.info("Initializing PaddleOCR detector...")
    detector = load_paddle_detector(
        use_gpu=use_gpu,
        use_angle_cls=det_config.get("use_angle_cls", True),
        lang=det_config.get("lang", "vi"),
    )
    logger.info("Initializing VietOCR recognizer...")
    recognizer = load_vietocr_model(config_name=rec_engine_config, use_gpu=rec_gpu)
    ocr_engine = make_ocr_engine_metadata(det_config, rec_config, rec_engine_config)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for jsonl_path_str in args.data_files:
        jsonl_path = Path(jsonl_path_str)
        if not jsonl_path.exists():
            logger.warning("Data file not found: %s", jsonl_path)
            continue

        samples = read_jsonl(jsonl_path)
        if args.limit is not None:
            samples = samples[: args.limit]

        for sample in tqdm(samples, desc=f"OCR Cache {jsonl_path.name}"):
            sample_id = str(sample.get("id") or "")
            image_path_str = sample.get("image_path")
            if not image_path_str:
                logger.warning("Missing image_path in sample %s", sample_id)
                error_count += 1
                continue

            image_path = Path(image_path_str)
            cache_path = cache_path_for_sample(sample, output_dir)
            if cache_path is None:
                cache_path = default_cache_dir / f"{_safe_stem(sample_id or image_path.stem)}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if cache_path.exists() and not args.overwrite:
                skipped_count += 1
                continue
            if not image_path.exists():
                logger.error("Image file not found: %s for sample %s", image_path, sample_id)
                error_count += 1
                continue

            try:
                start_total = time.time()
                image_hash = file_md5(image_path)
                image = load_receipt_image(image_path)
                pre = preprocess_receipt_image(
                    image,
                    profile=args.preprocess_profile,
                    max_long_side=args.max_long_side,
                )
                preprocessed_path = save_preprocessed_image(pre, sample_id or image_path.stem, preprocessed_dir)

                detect_start = time.time()
                regions = detect_text_regions(detector, str(preprocessed_path))
                detect_ms = (time.time() - detect_start) * 1000

                recognize_ms = 0.0
                if regions:
                    cropped_imgs = [crop_region(pre.image, r["bbox"], padding=2) for r in regions]
                    recognize_start = time.time()
                    texts = recognize_regions(recognizer, cropped_imgs, batch_size=args.batch_size)
                    recognize_ms = (time.time() - recognize_start) * 1000
                    for region, text in zip(regions, texts):
                        region["text"] = text.strip()
                    regions = [region for region in regions if region.get("text")]

                flat_words, grouped_lines = sort_reading_order(regions, y_threshold=y_threshold)
                cache_data = build_cache_base(
                    sample_id=sample_id,
                    image_path=image_path,
                    image_hash=image_hash,
                    ocr_engine=ocr_engine,
                    pre=pre,
                    cache_version=args.cache_version,
                    preprocessed_image_path=preprocessed_path,
                )
                cache_data.update({
                    "boxes": [word["bbox"] for word in flat_words],
                    "lines": [
                        [
                            {"bbox": w["bbox"], "polygon": w.get("polygon", []), "text": w["text"]}
                            for w in line
                        ]
                        for line in grouped_lines
                    ],
                    "words": [
                        {"bbox": w["bbox"], "polygon": w.get("polygon", []), "text": w["text"]}
                        for w in flat_words
                    ],
                    "latency": {
                        "detect_ms": detect_ms,
                        "recognize_ms": recognize_ms,
                        "total_ms": (time.time() - start_total) * 1000,
                    },
                })

                with open(cache_path, "w", encoding="utf-8") as out_f:
                    json.dump(cache_data, out_f, ensure_ascii=False, indent=2)
                processed_count += 1
            except Exception as exc:
                logger.error("Error processing sample %s: %s", sample_id, exc, exc_info=True)
                error_count += 1

    logger.info(
        "OCR caching completed: processed=%s skipped=%s errors=%s",
        processed_count,
        skipped_count,
        error_count,
    )


if __name__ == "__main__":
    main()
