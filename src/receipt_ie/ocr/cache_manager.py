import json
import hashlib
import os
import time
import yaml
from pathlib import Path
from PIL import Image

from receipt_ie.preprocessing.image_preprocess import preprocess_receipt_image
from receipt_ie.ocr.detect_paddle import detect_text_regions, crop_region
from receipt_ie.ocr.recognize_vietocr import recognize_regions
from receipt_ie.ocr.reading_order import sort_reading_order

def compute_image_hash(image: Image.Image) -> str:
    """
    Tính MD5 hash của PIL Image.
    """
    hash_md5 = hashlib.md5()
    # Chuyển ảnh về bytes và update hash
    hash_md5.update(image.tobytes())
    return hash_md5.hexdigest()

def get_or_build_ocr(
    image: Image.Image,
    detector,
    recognizer,
    preprocess_profile: str = "resize",
    max_long_side: int = 1600,
    ocr_config_path: str = "configs/ocr.yaml",
    cache_dir: str = "outputs/runtime_ocr_cache",
) -> dict:
    """
    Tìm kiếm OCR cache của ảnh. Nếu có sẵn, trả về.
    Nếu chưa có, chạy tiền xử lý, chạy OCR online, sắp xếp reading order, ghi cache JSON và trả về.
    """
    # 1. Tính toán lookup key
    image_hash = compute_image_hash(image)
    
    # Đọc cấu hình OCR để đưa vào lookup key
    det_key = "default"
    rec_key = "default"
    y_threshold = 12
    cache_version = "v2"
    
    if os.path.exists(ocr_config_path):
        try:
            with open(ocr_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            det_cfg = cfg.get("detection", {})
            rec_cfg = cfg.get("recognition", {})
            cache_cfg = cfg.get("cache", {})
            
            det_key = f"{det_cfg.get('gpu', False)}_{det_cfg.get('use_angle_cls', False)}_{det_cfg.get('lang', 'vi')}"
            rec_key = f"{rec_cfg.get('default_config', 'vgg_transformer')}_{rec_cfg.get('gpu', False)}"
            y_threshold = cache_cfg.get("reading_order_y_threshold", 12)
            cache_version = cache_cfg.get("version", "v2")
        except Exception:
            pass
            
    lookup_key = f"{image_hash}_{preprocess_profile}_{max_long_side}_{det_key}_{rec_key}_{cache_version}"
    
    cache_folder = Path(cache_dir)
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_path = cache_folder / f"{lookup_key}.json"
    
    # 2. Đọc từ cache nếu tồn tại
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    # 3. Chạy OCR online nếu chưa có cache
    start_total = time.time()
    
    # Tiền xử lý ảnh
    pre = preprocess_receipt_image(
        image,
        profile=preprocess_profile,
        max_long_side=max_long_side,
    )
    
    # Lưu tạm ảnh preprocessed ra đĩa vì PaddleOCR detector yêu cầu path
    temp_dir = cache_folder / "_temp_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_img_path = temp_dir / f"{lookup_key}_preprocessed.png"
    pre.image.save(temp_img_path)
    
    try:
        detect_start = time.time()
        regions = detect_text_regions(detector, str(temp_img_path))
        detect_ms = (time.time() - detect_start) * 1000
        
        recognize_ms = 0.0
        if regions:
            cropped_imgs = [crop_region(pre.image, r["bbox"], padding=2) for r in regions]
            recognize_start = time.time()
            texts = recognize_regions(recognizer, cropped_imgs, batch_size=16)
            recognize_ms = (time.time() - recognize_start) * 1000
            for region, text in zip(regions, texts):
                region["text"] = text.strip()
            regions = [region for region in regions if region.get("text")]
            
        flat_words, grouped_lines = sort_reading_order(regions, y_threshold=y_threshold)
        
        # Build base metadata
        width, height = pre.image.size
        ocr_data = {
            "id": lookup_key,
            "image_hash": image_hash,
            "preprocess": {
                "profile": preprocess_profile,
                "max_long_side": max_long_side,
                "scale_x": pre.scale_x,
                "scale_y": pre.scale_y,
            },
            "image_size": [width, height],
            "original_size": [pre.metadata["original_width"], pre.metadata["original_height"]],
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
            }
        }
        
        # Lưu vào cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=2)
            
        return ocr_data
    finally:
        # Dọn dẹp ảnh tạm
        if temp_img_path.exists():
            try:
                temp_img_path.unlink()
            except Exception:
                pass
