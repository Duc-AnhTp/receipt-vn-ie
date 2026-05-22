"""
Script chạy suy luận baseline Rule-based trên tập test.
Đọc từ OCR cache (hoặc chạy OCR trực tiếp nếu chưa có cache), trích xuất và lưu predictions.
"""
import torch
import os
import json
import time
import argparse
import logging
import yaml
import uuid
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from receipt_ie.baseline.rule_extractor import extract_fields_from_ocr
from receipt_ie.ocr.detect_paddle import load_paddle_detector, detect_text_regions, crop_region
from receipt_ie.ocr.recognize_vietocr import load_vietocr_model, recognize_regions
from receipt_ie.ocr.reading_order import sort_reading_order
from typing import Dict, Any
from receipt_ie.data.schemas import BaseExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EMPTY_FIELDS = {"store_name": "", "date": "", "total": "", "address": ""}


def _empty_fields() -> Dict[str, str]:
    return EMPTY_FIELDS.copy()


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except (PermissionError, OSError) as exc:
        logger.warning("Could not remove temporary OCR file %s: %s", path, exc)


class BaselineExtractor(BaseExtractor):
    """
    Bộ trích xuất Rule-based (Baseline) sử dụng PaddleOCR + VietOCR + Heuristics/Regex.
    Kế thừa interface BaseExtractor chung.
    """
    def __init__(
        self,
        detector=None,
        recognizer=None,
        ocr_config_path: str = "configs/ocr.yaml",
        project_root: str = "."
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.ocr_config_path = Path(project_root) / ocr_config_path
        self.project_root = Path(project_root)
        self.y_threshold = 12

    def load(self, checkpoint_path: str = "") -> None:
        """
        Khởi tạo các engine OCR nếu chưa có.
        """
        if self.detector is None or self.recognizer is None:
            self._init_ocr()

    def _init_ocr(self):
        if not self.ocr_config_path.exists():
            use_gpu = False
            rec_config = "vgg_transformer"
        else:
            with open(self.ocr_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            use_gpu = cfg.get("detection", {}).get("gpu", False)
            rec_config = cfg.get("recognition", {}).get("default_config", "vgg_transformer")
            self.y_threshold = cfg.get("cache", {}).get("reading_order_y_threshold", 12)
            
        if self.detector is None:
            self.detector = load_paddle_detector(use_gpu=use_gpu, use_angle_cls=False, lang="vi")
        if self.recognizer is None:
            rec_gpu = torch.cuda.is_available()
            self.recognizer = load_vietocr_model(config_name=rec_config, use_gpu=rec_gpu)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start_e2e = time.time()
        temp_img_path = None
        
        try:
            # Use a unique file name to avoid stale locks/collisions on Windows.
            temp_dir = self.project_root / "data/temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_img_path = temp_dir / f"baseline_{uuid.uuid4().hex}.png"
            image.save(temp_img_path)

            regions = detect_text_regions(self.detector, str(temp_img_path))
            if not regions:
                return {
                    "method": "baseline",
                    "prediction": _empty_fields(),
                    "normalized_prediction": _empty_fields(),
                    "raw_output": None,
                    "latency_cached_ms": 0.0,
                    "latency_e2e_ms": (time.time() - start_e2e) * 1000,
                    "status": "ok",
                    "error": None,
                    "words": []
                }
                
            cropped_imgs = [crop_region(image, r["bbox"], padding=2) for r in regions]
            texts = recognize_regions(self.recognizer, cropped_imgs, batch_size=16)
            for r, text in zip(regions, texts):
                r["text"] = text.strip()
            regions = [r for r in regions if r["text"]]
            
            if not regions:
                return {
                    "method": "baseline",
                    "prediction": _empty_fields(),
                    "normalized_prediction": _empty_fields(),
                    "raw_output": None,
                    "latency_cached_ms": 0.0,
                    "latency_e2e_ms": (time.time() - start_e2e) * 1000,
                    "status": "ok",
                    "error": None,
                    "words": []
                }
                
            flat_words, grouped_lines = sort_reading_order(regions, y_threshold=self.y_threshold)
            width, height = image.size
            ocr_data = {
                "image_size": [width, height],
                "lines": [
                    [{"bbox": w["bbox"], "polygon": w["polygon"], "text": w["text"]} for w in line]
                    for line in grouped_lines
                ],
                "words": [{"bbox": w["bbox"], "polygon": w["polygon"], "text": w["text"]} for w in flat_words]
            }
            
            start_rule = time.time()
            extracted = extract_fields_from_ocr(ocr_data)
            end_time = time.time()
            
            latency_cached = (end_time - start_rule) * 1000
            latency_e2e = (end_time - start_e2e) * 1000
            
            return {
                "method": "baseline",
                "prediction": extracted,
                "normalized_prediction": extracted,
                "raw_output": None,
                "latency_cached_ms": latency_cached,
                "latency_e2e_ms": latency_e2e,
                "status": "ok",
                "error": None,
                "words": flat_words
            }
        except Exception as exc:
            logger.exception("Baseline prediction failed")
            return {
                "method": "baseline",
                "prediction": _empty_fields(),
                "normalized_prediction": _empty_fields(),
                "raw_output": None,
                "latency_cached_ms": 0.0,
                "latency_e2e_ms": (time.time() - start_e2e) * 1000,
                "status": "error",
                "error": str(exc),
                "words": []
            }
        finally:
            if temp_img_path is not None:
                _cleanup_temp_file(temp_img_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy suy luận baseline Rule-based.")
    parser.add_argument(
        "--test_jsonl",
        type=str,
        default="data/processed/test.jsonl",
        help="Đường dẫn file test.jsonl"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="outputs/predictions/baseline_test.jsonl",
        help="Đường dẫn file JSONL đầu ra để lưu predictions"
    )
    parser.add_argument(
        "--config_ocr",
        type=str,
        default="configs/ocr.yaml",
        help="Đường dẫn file ocr.yaml"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số mẫu chạy test"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    test_file = Path(args.test_jsonl)
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
        
    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load config ocr
    ocr_config = load_config(args.config_ocr)
    cache_config = ocr_config.get("cache", {})
    y_threshold = cache_config.get("reading_order_y_threshold", 12)
    ocr_version = cache_config.get("version", "ocr_paddle27_vietocr_transformer_v1")
    
    # Lazy initialization cho OCR engines (chỉ khởi tạo khi có sample thiếu cache)
    detector = None
    recognizer = None
    
    def get_ocr_engines():
        nonlocal detector, recognizer
        if detector is None or recognizer is None:
            logger.info("OCR cache missing for some files. Initializing OCR engines for online fallback...")
            det_config = ocr_config.get("detection", {})
            rec_config = ocr_config.get("recognition", {})
            
            detector = load_paddle_detector(
                use_gpu=det_config.get("gpu", True),
                use_angle_cls=det_config.get("use_angle_cls", True),
                lang=det_config.get("lang", "vi")
            )
            recognizer = load_vietocr_model(
                config_name=rec_config.get("default_config", "vgg_transformer"),
                use_gpu=rec_config.get("gpu", True)
            )
        return detector, recognizer

    # Đọc test samples
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
                
    if args.limit is not None:
        samples = samples[:args.limit]
        
    logger.info(f"Running baseline inference on {len(samples)} samples...")
    
    count = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="Baseline Inference"):
            sample_id = sample.get("id")
            image_path_str = sample.get("image_path")
            ocr_cache_path_str = sample.get("ocr_cache_path")
            
            prediction_record = {
                "id": sample_id,
                "method": "baseline",
                "prediction": {},
                "normalized_prediction": {},
                "raw_output": None,
                "latency_cached_ms": 0.0,
                "latency_e2e_ms": 0.0,
                "status": "ok",
                "error": None
            }
            
            try:
                ocr_data = None
                ocr_start_time = 0
                ocr_end_time = 0
                
                # Thử đọc từ OCR cache trước
                if ocr_cache_path_str and os.path.exists(ocr_cache_path_str):
                    with open(ocr_cache_path_str, "r", encoding="utf-8") as cache_f:
                        ocr_data = json.load(cache_f)
                else:
                    # Chạy OCR online nếu chưa có cache
                    if not image_path_str or not os.path.exists(image_path_str):
                        raise FileNotFoundError(f"Image not found at {image_path_str} and cache missing.")
                        
                    det_eng, rec_eng = get_ocr_engines()
                    
                    ocr_start_time = time.time()
                    # 1. Detect
                    img = Image.open(image_path_str).convert("RGB")
                    width, height = img.size
                    regions = detect_text_regions(det_eng, image_path_str)
                    
                    if regions:
                        # 2. Crop & Recognize
                        cropped_imgs = [crop_region(img, r["bbox"], padding=2) for r in regions]
                        texts = recognize_regions(rec_eng, cropped_imgs, batch_size=16)
                        
                        for r, text in zip(regions, texts):
                            r["text"] = text.strip()
                        regions = [r for r in regions if r["text"]]
                        
                        # 3. Reading order
                        flat_words, grouped_lines = sort_reading_order(regions, y_threshold=y_threshold)
                        ocr_data = {
                            "id": sample_id,
                            "ocr_engine": ocr_version,
                            "image_size": [width, height],
                            "lines": [
                                [{"bbox": w["bbox"], "polygon": w["polygon"], "text": w["text"]} for w in line]
                                for line in grouped_lines
                            ],
                            "words": [{"bbox": w["bbox"], "polygon": w["polygon"], "text": w["text"]} for w in flat_words]
                        }
                    else:
                        ocr_data = {"lines": [], "words": []}
                    ocr_end_time = time.time()
                
                # Đo lường thời gian trích xuất heuristic
                rule_start_time = time.time()
                extracted = extract_fields_from_ocr(ocr_data)
                rule_end_time = time.time()
                
                # Tính toán Latency
                latency_cached_ms = (rule_end_time - rule_start_time) * 1000.0
                ocr_latency_ms = (ocr_end_time - ocr_start_time) * 1000.0 if ocr_start_time > 0 else 0.0
                latency_e2e_ms = latency_cached_ms + ocr_latency_ms
                
                # Cập nhật kết quả
                prediction_record["prediction"] = extracted
                # Baseline tự động chuẩn hóa các trường khi trích xuất
                prediction_record["normalized_prediction"] = extracted
                prediction_record["latency_cached_ms"] = round(latency_cached_ms, 2)
                prediction_record["latency_e2e_ms"] = round(latency_e2e_ms, 2)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {sample_id}: {e}")
                prediction_record["status"] = "error"
                prediction_record["error"] = str(e)
                
            out_f.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            count += 1
            
    logger.info(f"Baseline inference completed. Saved {count} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
