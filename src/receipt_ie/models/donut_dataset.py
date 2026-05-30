import json
import os
from pathlib import Path
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import DonutProcessor

from receipt_ie.data.build_donut_dataset import target_to_donut_sequence
from receipt_ie.data.augmentation import get_donut_transforms, apply_transforms

class DonutDataset(Dataset):
    """
    Custom PyTorch Dataset cho mô hình Donut.
    Đọc từ file unified JSONL, nạp ảnh và sinh chuỗi token mục tiêu cho decoder.
    """
    def __init__(
        self,
        jsonl_path: str,
        processor: DonutProcessor,
        task_token: str = "<s_receipt_ie>",
        max_length: int = 192,
        project_root: str = ".",
        is_train: bool = False,
        strict_image: bool = True
    ):
        self.processor = processor
        self.task_token = task_token
        self.max_length = max_length
        self.project_root = Path(project_root)
        self.is_train = is_train
        self.strict_image = strict_image
        
        if self.is_train:
            self.transform = get_donut_transforms()
            
        self.samples: List[Dict[str, Any]] = []
        
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.is_absolute():
            jsonl_file = self.project_root / jsonl_file
            
        if not jsonl_file.exists():
            print(f"Cảnh báo: Không tìm thấy file JSONL tại {jsonl_file}")
            return
            
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"Đã load {len(self.samples)} mẫu dữ liệu Donut từ {jsonl_path} (is_train={self.is_train})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # 1. Tải và xử lý ảnh
        img_rel_path = sample["image_path"]
        img_path = self.project_root / img_rel_path
        
        # Fallback: nếu ảnh raw không tồn tại, tìm ảnh đã tiền xử lý (resize)
        if not img_path.exists():
            img_stem = Path(img_rel_path).stem  # Ví dụ: mcocr_public_145013rpxpi hoặc img_399
            # Tìm trong thư mục preprocessed_images/resize với các pattern có thể có
            resize_dir = self.project_root / "data" / "interim" / "preprocessed_images" / "resize"
            found = False
            if resize_dir.exists():
                for candidate in resize_dir.iterdir():
                    if img_stem in candidate.stem:
                        img_path = candidate
                        found = True
                        break
            if not found and self.strict_image:
                raise FileNotFoundError(img_path)
            # Tạo một ảnh mock trắng để tránh crash khi training
            image = Image.new("RGB", (self.processor.image_processor.size["width"], 
                                      self.processor.image_processor.size["height"]), color="white")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
                if self.is_train:
                    image = apply_transforms(image, self.transform)
            except Exception as e:
                if self.strict_image:
                    raise RuntimeError(f"Could not load image {img_path}: {e}") from e
                print(f"Lỗi đọc ảnh {img_path}: {e}")
                image = Image.new("RGB", (self.processor.image_processor.size["width"], 
                                          self.processor.image_processor.size["height"]), color="white")

        # Tiền xử lý ảnh qua processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]
        
        # 2. Xây dựng target text sequence và token hóa
        target = sample["target"]
        target_sequence = target_to_donut_sequence(target, self.task_token)
        
        # Token hóa target text sequence
        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False, # Đã có sẵn task_token ở đầu và kết thúc rồi
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        # Thay thế pad_token_id bằng -100 để không tính loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "id": sample["id"],
            "target_sequence": target_sequence
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function cho DataLoader ghép các sample thành batch.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }
