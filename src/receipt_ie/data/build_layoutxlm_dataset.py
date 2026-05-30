import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, LayoutLMv2ImageProcessor
from PIL import Image

from receipt_ie.data.build_layoutxlm_labels import assign_word_labels, align_tokens_layoutxlm
from receipt_ie.ocr.reading_order import sort_reading_order
from receipt_ie.data.augmentation import get_layoutxlm_transforms, apply_transforms

logger = logging.getLogger(__name__)


def _scale_box(box: List[int], scale_x: float, scale_y: float) -> List[int]:
    return [
        int(round(box[0] * scale_x)),
        int(round(box[1] * scale_y)),
        int(round(box[2] * scale_x)),
        int(round(box[3] * scale_y)),
    ]


def _scale_field_boxes(
    field_boxes: Dict[str, List[List[int]]],
    scale_x: float,
    scale_y: float
) -> Dict[str, List[List[int]]]:
    return {
        field: [_scale_box(box, scale_x, scale_y) for box in boxes]
        for field, boxes in field_boxes.items()
    }


class LayoutXLMDataset(Dataset):
    """
    Custom Dataset cho LayoutXLM (đọc từ unified JSONL).
    Hỗ trợ hai chế độ nạp dữ liệu:
    - 'ocr_cache': Đọc kết quả detect+recognize từ file OCR cache cục bộ.
    - 'oracle_ocr': Đọc trực tiếp ground-truth OCR (texts & bboxes) từ dữ liệu gốc.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        mode: str = "ocr_cache",
        max_length: int = 512,
        project_root: str = ".",
        annotation_level_filter: str = "json_and_boxes",
        is_train: bool = False,
        overlap_threshold: float = 0.5
    ):
        self.tokenizer = tokenizer
        self.mode = mode.lower()
        self.max_length = max_length
        self.project_root = Path(project_root)
        self.annotation_level_filter = annotation_level_filter
        self.is_train = is_train
        self.overlap_threshold = overlap_threshold
        
        self.transform = None
        if self.is_train:
            self.transform = get_layoutxlm_transforms()
            
        # Khởi tạo Image Processor cho LayoutXLM
        try:
            model_path = tokenizer.name_or_path if hasattr(tokenizer, "name_or_path") else "microsoft/layoutxlm-base"
            self.image_processor = LayoutLMv2ImageProcessor.from_pretrained(model_path)
        except Exception:
            self.image_processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutxlm-base")
            
        self.samples: List[Dict[str, Any]] = []
        
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.is_absolute():
            jsonl_file = self.project_root / jsonl_file
            
        if not jsonl_file.exists():
            logger.warning(f"Không tìm thấy file JSONL tại {jsonl_file}")
            return
            
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line.strip())
                
                # Lọc dữ liệu theo annotation_level
                if self.annotation_level_filter:
                    if sample.get("annotation_level") != self.annotation_level_filter:
                        continue
                        
                # Nếu là chế độ oracle_ocr, yêu cầu phải có trường oracle_ocr
                if self.mode == "oracle_ocr":
                    if "oracle_ocr" not in sample or not sample["oracle_ocr"]:
                        continue
                        
                self.samples.append(sample)
                
        logger.info(f"Đã load {len(self.samples)} mẫu LayoutXLM (mode: {self.mode}) từ {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        
        width = sample["width"]
        height = sample["height"]
        img_rel_path = sample.get("image_path")
        cache_data: Dict[str, Any] = {}
        
        words: List[str] = []
        word_boxes: List[List[int]] = []
        
        if self.mode == "ocr_cache":
            # Nạp từ file OCR cache
            cache_rel_path = sample.get("ocr_cache_path")
            if not cache_rel_path:
                raise ValueError(f"Mẫu {sample_id} thiếu ocr_cache_path")
                
            cache_path = self.project_root / cache_rel_path
            if not cache_path.exists():
                # Ném lỗi cảnh báo rõ ràng nếu thiếu cache khi train
                raise FileNotFoundError(f"Không tìm thấy file OCR cache tại: {cache_path}. Vui lòng chạy build_ocr_cache trước.")
                
            with open(cache_path, "r", encoding="utf-8") as cf:
                cache_data = json.load(cf)
                
            # Đọc danh sách words đã được sắp xếp từ cache
            cache_words = cache_data.get("words", [])
            words = [w["text"] for w in cache_words]
            word_boxes = [w["bbox"] for w in cache_words]

            preprocessed_image_path = cache_data.get("preprocessed_image_path")
            if preprocessed_image_path:
                img_rel_path = preprocessed_image_path

            cache_size = cache_data.get("preprocessed_size") or cache_data.get("image_size")
            if cache_size and len(cache_size) == 2:
                width, height = int(cache_size[0]), int(cache_size[1])
            
        elif self.mode == "oracle_ocr":
            # Nạp từ ground-truth OCR trong sample
            oracle_ocr_data = sample.get("oracle_ocr", [])
            
            # Sắp xếp thứ tự đọc tự nhiên cho oracle ocr
            regions = [{"bbox": w["box"], "text": w["text"]} for w in oracle_ocr_data]
            sorted_words, _ = sort_reading_order(regions, y_threshold=12)
            
            words = [w["text"] for w in sorted_words]
            word_boxes = [w["bbox"] for w in sorted_words]
            
        # 1. Gán nhãn BIO viết hoa cho từng word dựa trên overlap với field_boxes
        field_boxes = sample.get("field_boxes", {})
        preprocess_info = cache_data.get("preprocess", {})
        preprocess_meta = preprocess_info.get("metadata", {}) if isinstance(preprocess_info, dict) else {}
        original_size = preprocess_info.get("original_size") or preprocess_meta.get("original_size") or cache_data.get("original_size")
        if not original_size and preprocess_meta.get("original_width") and preprocess_meta.get("original_height"):
            original_size = [preprocess_meta["original_width"], preprocess_meta["original_height"]]
        processed_size = (
            preprocess_info.get("processed_size")
            or cache_data.get("preprocessed_size")
        )
        coordinate_transform = (
            preprocess_info.get("coordinate_transform")
            or preprocess_meta.get("coordinate_transform")
            or cache_data.get("coordinate_transform")
        )
        if (
            self.mode == "ocr_cache"
            and coordinate_transform == "scale"
            and original_size
            and processed_size
            and len(original_size) == 2
            and len(processed_size) == 2
        ):
            scale_x = processed_size[0] / original_size[0] if original_size[0] else 1.0
            scale_y = processed_size[1] / original_size[1] if original_size[1] else 1.0
            field_boxes = _scale_field_boxes(field_boxes, scale_x, scale_y)
        word_labels = assign_word_labels(words, word_boxes, field_boxes, overlap_threshold=self.overlap_threshold)
        
        # 2. Tokenize và alignment tokens/boxes/labels
        aligned = align_tokens_layoutxlm(
            words=words,
            word_boxes=word_boxes,
            word_labels=word_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            image_size=(width, height)
        )
        
        # 3. Tải và xử lý ảnh (bắt buộc đối với LayoutLMv2 / LayoutXLM)
        img_path = self.project_root / img_rel_path if img_rel_path else Path("")
        
        # Fallback nếu ảnh không tồn tại
        if not img_path.exists() or not img_path.is_file():
            image = Image.new("RGB", (224, 224), color="white")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
                if self.is_train:
                    image = apply_transforms(image, self.transform)
            except Exception as e:
                logger.warning(f"Lỗi đọc ảnh {img_path}: {e}")
                image = Image.new("RGB", (224, 224), color="white")
                
        # Tiền xử lý ảnh qua processor
        pixel_values = self.image_processor(image, apply_ocr=False, return_tensors="pt").pixel_values[0]
        
        # Trả về định dạng tensor PyTorch
        return {
            "input_ids": torch.tensor(aligned["input_ids"], dtype=torch.long),
            "bbox": torch.tensor(aligned["bbox"], dtype=torch.long),
            "attention_mask": torch.tensor(aligned["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned["labels"], dtype=torch.long),
            "image": pixel_values,
            "id": sample_id
        }

def layoutxlm_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function ghép các sample LayoutXLM thành batch.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    bbox = torch.stack([item["bbox"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "bbox": bbox,
        "attention_mask": attention_mask,
        "labels": labels,
        "image": images
    }
