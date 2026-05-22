import json
import argparse
import ast
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from receipt_ie.data.normalize_text import (
    normalize_vietnamese_text,
    normalize_store_name,
    normalize_address,
    normalize_date,
    normalize_money
)

# Mapping nhãn từ MC-OCR sang unified schema
LABEL_MAP = {
    "SELLER": "store_name",
    "SELLER_NAME": "store_name",
    "SELLER_ADDRESS": "address",
    "ADDRESS": "address",
    "TIMESTAMP": "date",
    "TIMESTAMPS": "date",
    "TOTAL_COST": "total"
}

def normalize_field(field: str, value: str) -> str:
    if field == "store_name":
        return normalize_store_name(value)
    elif field == "address":
        return normalize_address(value)
    elif field == "date":
        return normalize_date(value)
    elif field == "total":
        return normalize_money(value)
    return normalize_vietnamese_text(value)

def parse_polygons(raw_val) -> list:
    """
    Parse chuỗi polygon từ cột anno_polygons.
    Format có thể là list các toạ độ hoặc chuỗi dạng list.
    """
    if pd.isna(raw_val) or not isinstance(raw_val, str) or not raw_val.strip():
        return []
    try:
        # anno_polygons chứa list các list toạ độ: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
        return ast.literal_eval(raw_val)
    except Exception:
        return []

def safe_split(val, delimiter="|||") -> list:
    """
    Tách chuỗi theo dấu delimiter và loại bỏ khoảng trắng.
    """
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return []
    return [x.strip() for x in val.split(delimiter) if x.strip()]

def polygon_to_bbox(poly: list) -> list:
    """
    Chuyển đổi một polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] sang bbox [x0, y0, x1, y1].
    """
    if not poly or len(poly) < 3:
        return []
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

def convert_mcocr(csv_path: str, images_dir: str, output_jsonl: str, project_root: str = "."):
    """
    Convert MC-OCR 2021 train.csv sang unified JSONL.
    Lưu image_path dạng tương đối so với project_root.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Không tìm thấy file CSV tại {csv_path}. Vui lòng kiểm tra lại dữ liệu.")
        return
        
    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    print(f"Đọc thành công CSV với {len(df)} dòng.")
    
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_id = row["img_id"]
            img_path = Path(images_dir) / img_id
            
            # Bỏ qua dòng không có annotation nhãn (ví dụ MC-OCR public test)
            raw_labels = row.get("anno_labels", "")
            if pd.isna(raw_labels) or not str(raw_labels).strip():
                continue
            
            if not img_path.exists():
                # Bỏ qua nếu không tìm thấy ảnh
                continue
                
            # Đọc kích thước ảnh
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                continue
            
            # Parse các trường dữ liệu
            texts = safe_split(row.get("anno_texts", ""))
            labels = safe_split(row.get("anno_labels", ""))
            polygons = parse_polygons(row.get("anno_polygons", ""))
            
            raw_target = {
                "store_name": "",
                "date": "",
                "total": "",
                "address": ""
            }
            
            target = {
                "store_name": "",
                "date": "",
                "total": "",
                "address": ""
            }
            
            field_boxes = {
                "store_name": [],
                "date": [],
                "total": [],
                "address": []
            }
            
            field_polygons = {
                "store_name": [],
                "date": [],
                "total": [],
                "address": []
            }
            
            # Ghép nối các dòng text cùng label
            # Đồng thời gán box và polygon tương ứng
            # Lưu ý: len(texts) và len(labels) phải khớp nhau
            for t_idx, (text, label) in enumerate(zip(texts, labels)):
                label_upper = label.strip().upper()
                unified_field = LABEL_MAP.get(label_upper)
                
                if unified_field is None:
                    continue
                
                # Cập nhật raw target
                if raw_target[unified_field]:
                    raw_target[unified_field] += " " + text
                else:
                    raw_target[unified_field] = text
                
                # Cập nhật normalized target
                norm_val = normalize_field(unified_field, text)
                if norm_val:
                    if target[unified_field]:
                        # Chỉ ghép tiền hoặc ngày nếu thật sự cần thiết (ở đây dùng dấu cách)
                        target[unified_field] += " " + norm_val
                    else:
                        target[unified_field] = norm_val
                
                # Gán polygon và box nếu có tọa độ tương ứng
                if t_idx < len(polygons):
                    poly = polygons[t_idx]
                    bbox = polygon_to_bbox(poly)
                    if bbox:
                        field_boxes[unified_field].append(bbox)
                        field_polygons[unified_field].append(poly)
                        
            # Thu thập toàn bộ ground-truth OCR cho chế độ oracle_ocr
            oracle_ocr = []
            for t_idx, text in enumerate(texts):
                if t_idx < len(polygons):
                    poly = polygons[t_idx]
                    bbox = polygon_to_bbox(poly)
                    if bbox:
                        oracle_ocr.append({
                            "text": text,
                            "box": bbox
                        })

            # MC-OCR có đầy đủ nhãn và polygon
            sample = {
                "id": Path(img_id).stem,
                # Gom nhóm theo cửa hàng nếu có thể đoán qua store_name, 
                # tạm thời dùng tên ảnh làm group_id vì MC-OCR không cung cấp group_id rõ ràng
                "group_id": Path(img_id).stem,
                "store_group": normalize_store_name(raw_target["store_name"]) or "mcocr_unknown",
                "source": "mc_ocr_2021",
                "image_path": os.path.relpath(img_path, project_root).replace("\\", "/"),
                "width": width,
                "height": height,
                "annotation_level": "json_and_boxes",
                "raw_target": raw_target,
                "target": target,
                "field_boxes": field_boxes,
                "field_polygons": field_polygons,
                "oracle_ocr": oracle_ocr,
                "ocr_cache_path": f"data/interim/ocr_cache/{Path(img_id).stem}.json"
            }
            
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Đã chuyển đổi thành công {count} mẫu MC-OCR có nhãn sang {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi MC-OCR 2021 sang unified JSONL")
    parser.add_argument("--csv_path", type=str, default="data/raw/mc_ocr_2021/mcocr_train_df.csv", help="Đường dẫn file CSV")
    parser.add_argument("--images_dir", type=str, default="data/raw/mc_ocr_2021/train_images", help="Thư mục ảnh")
    parser.add_argument("--output_jsonl", type=str, default="data/interim/mcocr_unified.jsonl", help="File JSONL đầu ra")
    parser.add_argument("--project_root", type=str, default=".", help="Gốc project để tính đường dẫn tương đối")
    args = parser.parse_args()
    
    convert_mcocr(args.csv_path, args.images_dir, args.output_jsonl, args.project_root)
