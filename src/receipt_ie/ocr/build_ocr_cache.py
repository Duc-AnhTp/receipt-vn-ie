"""
Script offline chạy PaddleOCR detect + VietOCR recognize để tạo OCR cache.
Lưu kết quả dưới dạng JSON cho từng ảnh, sắp xếp theo thứ tự đọc tự nhiên.
"""
import os
import json
import argparse
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from receipt_ie.ocr.detect_paddle import load_paddle_detector, detect_text_regions, crop_region
from receipt_ie.ocr.recognize_vietocr import load_vietocr_model, recognize_regions
from receipt_ie.ocr.reading_order import sort_reading_order

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy OCR offline và tạo cache.")
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=["data/interim/mcocr_unified.jsonl", "data/interim/self_unified.jsonl"],
        help="Danh sách các file JSONL unified cần chạy OCR cache."
    )
    parser.add_argument(
        "--config_ocr",
        type=str,
        default="configs/ocr.yaml",
        help="Đường dẫn đến file cấu hình ocr.yaml."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè file cache đã tồn tại."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng ảnh xử lý (dùng để test)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size cho VietOCR recognizer."
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # Load config
    if not os.path.exists(args.config_ocr):
        logger.error(f"Config file not found: {args.config_ocr}")
        return
        
    config = load_config(args.config_ocr)
    
    # Trích xuất cấu hình
    det_config = config.get("detection", {})
    rec_config = config.get("recognition", {})
    cache_config = config.get("cache", {})
    
    use_gpu = det_config.get("gpu", True)
    use_angle_cls = det_config.get("use_angle_cls", True)
    lang = det_config.get("lang", "vi")
    
    rec_engine_config = rec_config.get("default_config", "vgg_transformer")
    rec_gpu = rec_config.get("gpu", True)
    
    cache_dir = Path(cache_config.get("dir", "data/interim/ocr_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    y_threshold = cache_config.get("reading_order_y_threshold", 12)
    ocr_version = cache_config.get("version", "ocr_paddle27_vietocr_transformer_v1")
    
    # Khởi tạo mô hình
    logger.info("Initializing PaddleOCR Detector...")
    detector = load_paddle_detector(use_gpu=use_gpu, use_angle_cls=use_angle_cls, lang=lang)
    
    logger.info("Initializing VietOCR Recognizer...")
    recognizer = load_vietocr_model(config_name=rec_engine_config, use_gpu=rec_gpu)
    
    # Duyệt qua các file dữ liệu
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for jsonl_path_str in args.data_files:
        jsonl_path = Path(jsonl_path_str)
        if not jsonl_path.exists():
            logger.warning(f"Data file not found: {jsonl_path}")
            continue
            
        logger.info(f"Processing data file: {jsonl_path}")
        
        # Đọc tất cả các dòng
        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
                    
        # Áp dụng limit nếu có
        if args.limit is not None:
            samples = samples[:args.limit]
            
        for sample in tqdm(samples, desc=f"OCR Cache {jsonl_path.name}"):
            sample_id = sample.get("id")
            image_path_str = sample.get("image_path")
            ocr_cache_path_str = sample.get("ocr_cache_path")
            
            if not image_path_str or not ocr_cache_path_str:
                logger.warning(f"Missing image_path or ocr_cache_path in sample {sample_id}")
                continue
                
            image_path = Path(image_path_str)
            ocr_cache_path = Path(ocr_cache_path_str)
            
            # Đảm bảo thư mục cha của ocr_cache_path tồn tại
            ocr_cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Kiểm tra xem cache đã tồn tại chưa
            if ocr_cache_path.exists() and not args.overwrite:
                skipped_count += 1
                continue
                
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path} for sample {sample_id}")
                error_count += 1
                continue
                
            try:
                # Đọc ảnh
                img = Image.open(image_path).convert("RGB")
                width, height = img.size
                
                # 1. Phát hiện vùng chữ (PaddleOCR)
                regions = detect_text_regions(detector, str(image_path))
                
                if not regions:
                    # Nếu không phát hiện thấy chữ nào, lưu cache trống
                    cache_data = {
                        "id": sample_id,
                        "ocr_engine": ocr_version,
                        "preprocess_version": "v1",
                        "image_size": [width, height],
                        "lines": [],
                        "words": []
                    }
                    with open(ocr_cache_path, "w", encoding="utf-8") as out_f:
                        json.dump(cache_data, out_f, ensure_ascii=False, indent=2)
                    processed_count += 1
                    continue
                
                # 2. Cắt ảnh và nhận dạng chữ (VietOCR)
                cropped_imgs = []
                for r in regions:
                    crop_img = crop_region(img, r["bbox"], padding=2)
                    cropped_imgs.append(crop_img)
                    
                texts = recognize_regions(recognizer, cropped_imgs, batch_size=args.batch_size)
                
                # Gán text nhận dạng được vào region tương ứng
                for r, text in zip(regions, texts):
                    r["text"] = text.strip()
                    
                # Lọc bỏ các vùng nhận dạng ra chuỗi rỗng
                regions = [r for r in regions if r["text"]]
                
                # 3. Sắp xếp thứ tự đọc (Reading Order)
                flat_words, grouped_lines = sort_reading_order(regions, y_threshold=y_threshold)
                
                # Chuẩn bị dữ liệu lưu cache
                cache_data = {
                    "id": sample_id,
                    "ocr_engine": ocr_version,
                    "preprocess_version": "v1",
                    "image_size": [width, height],
                    "lines": [
                        [
                            {
                                "bbox": w["bbox"],
                                "polygon": w["polygon"],
                                "text": w["text"]
                            }
                            for w in line
                        ]
                        for line in grouped_lines
                    ],
                    "words": [
                        {
                            "bbox": w["bbox"],
                            "polygon": w["polygon"],
                            "text": w["text"]
                        }
                        for w in flat_words
                    ]
                }
                
                # Ghi ra file cache
                with open(ocr_cache_path, "w", encoding="utf-8") as out_f:
                    json.dump(cache_data, out_f, ensure_ascii=False, indent=2)
                    
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}", exc_info=True)
                error_count += 1
                
    logger.info(f"OCR caching completed: Processed {processed_count}, Skipped {skipped_count}, Errors {error_count}")


if __name__ == "__main__":
    main()
