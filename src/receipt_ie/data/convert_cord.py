import json
import argparse
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
import sys

# Thêm src vào path để import
sys.path.append(str(Path(__file__).parent.parent.parent))

from receipt_ie.data.normalize_text import normalize_money

def extract_cord_total(ground_truth_str: str) -> str:
    """
    Trích xuất tổng tiền từ ground_truth của CORD v2.
    CORD v2 format ground_truth là một chuỗi JSON.
    """
    try:
        gt = json.loads(ground_truth_str)
        parse_data = gt.get("gt_parse", {})
        
        # total_price có thể nằm trong field total hoặc sub_total
        total_obj = parse_data.get("total", {}) or {}
        total_price = total_obj.get("total_price", "")
        
        if not total_price:
            subtotal_obj = parse_data.get("sub_total", {}) or {}
            total_price = subtotal_obj.get("subtotal_price", "")
            
        return normalize_money(total_price)
    except Exception:
        return ""

def convert_cord(cord_dir: str, output_jsonl: str, images_dir: str):
    """
    Đọc CORD v2 và ghi nhận ảnh cùng nhãn total đã được chuẩn hóa.
    Chỉ lấy tập 'train' và 'validation' để warm-up.
    """
    cord_path = Path(cord_dir)
    if not cord_path.exists():
        print(f"Không tìm thấy dữ liệu CORD v2 tại {cord_dir}. Vui lòng chạy download_cord.py trước.")
        return
        
    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    img_out_dir = Path(images_dir)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_from_disk(str(cord_path))
    
    # Ghi nhận tập train và validation để làm warmup
    splits = ["train", "validation"]
    
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for split in splits:
            if split not in dataset:
                continue
            
            print(f"Đang xử lý split '{split}' của CORD v2...")
            for idx, ex in enumerate(tqdm(dataset[split])):
                image = ex["image"]
                width, height = image.size
                
                # Lưu ảnh ra đĩa local
                img_name = f"cord_{split}_{idx:06d}.jpg"
                img_path = img_out_dir / img_name
                image.convert("RGB").save(img_path)
                
                # Trích xuất nhãn total
                total = extract_cord_total(ex["ground_truth"])
                
                sample = {
                    "id": f"cord_{split}_{idx:06d}",
                    "group_id": f"cord_group_{split}_{idx:06d}",
                    "store_group": "cord_unknown",
                    "source": "cord_v2",
                    "image_path": str(img_path.as_posix()),
                    "width": width,
                    "height": height,
                    "annotation_level": "json_only",
                    "raw_target": {
                        "store_name": "",
                        "date": "",
                        "total": "", # Không lưu raw vì ta lấy trực tiếp qua parse
                        "address": ""
                    },
                    "target": {
                        "store_name": "",
                        "date": "",
                        "total": total,
                        "address": ""
                    },
                    "field_boxes": {
                        "store_name": [],
                        "date": [],
                        "total": [],
                        "address": []
                    },
                    "field_polygons": {
                        "store_name": [],
                        "date": [],
                        "total": [],
                        "address": []
                    },
                    "ocr_cache_path": ""
                }
                
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"Đã lưu {count} mẫu CORD v2 warm-up thành công vào {output_jsonl}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi CORD v2 sang định dạng unified JSONL cho Donut Warm-up")
    parser.add_argument("--cord_dir", type=str, default="data/raw/cord_v2", help="Đường dẫn dữ liệu CORD v2")
    parser.add_argument("--output_jsonl", type=str, default="data/processed/donut/cord_warmup_train.jsonl", help="Đường dẫn file JSONL đầu ra")
    parser.add_argument("--images_dir", type=str, default="data/interim/cord_v2_images", help="Thư mục lưu ảnh CORD v2")
    args = parser.parse_args()
    
    convert_cord(args.cord_dir, args.output_jsonl, args.images_dir)
