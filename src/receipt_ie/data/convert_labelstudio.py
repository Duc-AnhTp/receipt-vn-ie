import json
import argparse
import os
from pathlib import Path
from PIL import Image

from receipt_ie.data.normalize_text import (
    normalize_store_name,
    normalize_address,
    normalize_date,
    normalize_money,
    normalize_vietnamese_text
)

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

def convert_labelstudio(json_path: str, images_dir: str, output_jsonl: str, project_root: str = "."):
    """
    Convert file JSON xuất ra từ Label Studio sang unified JSONL.
    Lưu image_path dạng tương đối so với project_root.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Không tìm thấy file JSON của Label Studio tại {json_path}. Bỏ qua convert hoặc tạo file rỗng.")
        # Tạo file rỗng để không block các bước sau
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            pass
        return
        
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        
    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for task in data:
            # Lấy tên ảnh gốc
            # Label Studio thường lưu image URL hoặc path trong task['data']['image']
            image_url = task.get("data", {}).get("image", "")
            if not image_url:
                continue
                
            # Trích xuất filename từ url/path
            from urllib.parse import unquote
            decoded_url = unquote(image_url)
            filename = os.path.basename(decoded_url.replace("\\", "/"))
            
            # Xóa các hash prefix mà Label Studio sinh ra (ví dụ: b8a927c3-image.jpg -> image.jpg)
            # Label Studio thường thêm 8-10 ký tự hash đầu và dấu gạch ngang
            if "-" in filename and len(filename.split("-")[0]) == 8:
                filename = "-".join(filename.split("-")[1:])
                
            img_path = Path(images_dir) / filename
            if not img_path.exists():
                # Thử tìm với filename gốc (giữ cả hash)
                img_path = Path(images_dir) / os.path.basename(decoded_url.replace("\\", "/"))
                if not img_path.exists():
                    # Nếu vẫn không thấy, bỏ qua
                    continue
            
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                continue
                
            raw_target = {"store_name": "", "date": "", "total": "", "address": ""}
            target = {"store_name": "", "date": "", "total": "", "address": ""}
            field_boxes = {"store_name": [], "date": [], "total": [], "address": []}
            field_polygons = {"store_name": [], "date": [], "total": [], "address": []}
            
            # Parse annotations
            annotations = task.get("annotations", [])
            if not annotations:
                continue
                
            results = annotations[0].get("result", [])
            
            # Label Studio lưu tách biệt: 
            # 1. Bounding box có rectanglelabels
            # 2. Transcription (TextArea) có parent_id trỏ tới box hoặc cùng id
            # Ta sẽ gom nhóm theo id của box
            boxes_info = {}
            textareas = []
            texts_by_id = {}
            texts_by_parent = {}
            
            for res in results:
                res_type = res.get("type")
                res_id = res.get("id")
                from_name = res.get("from_name", "")
                
                if res_type == "rectanglelabels":
                    val = res.get("value", {})
                    labels = val.get("rectanglelabels", [])
                    if labels:
                        # Tỉ lệ % từ 0->100
                        x = val.get("x", 0)
                        y = val.get("y", 0)
                        w = val.get("width", 0)
                        h = val.get("height", 0)
                        
                        x0 = int(x * width / 100)
                        y0 = int(y * height / 100)
                        x1 = int((x + w) * width / 100)
                        y1 = int((y + h) * height / 100)
                        
                        boxes_info[res_id] = {
                            "label": labels[0],
                            "bbox": [x0, y0, x1, y1],
                            "poly": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                        }
                elif res_type == "textarea":
                    val = res.get("value", {})
                    texts = val.get("text", [])
                    if texts:
                        text = texts[0]
                        parent_id = res.get("parentID") or res.get("parent_id")
                        textarea = {
                            "id": res_id,
                            "parent_id": parent_id,
                            "from_name": from_name,
                            "to_name": res.get("to_name", ""),
                            "text": text
                        }
                        textareas.append(textarea)
                        if res_id:
                            texts_by_id[res_id] = textarea
                        if parent_id:
                            texts_by_parent[parent_id] = textarea
                        
            # Kết hợp Box và Text
            linked_textarea_ids = set()
            for box_id, box_data in boxes_info.items():
                label = box_data["label"]
                bbox = box_data["bbox"]
                poly = box_data["poly"]
                
                # Tìm text tương ứng
                textarea = texts_by_parent.get(box_id) or texts_by_id.get(box_id)
                text = textarea["text"] if textarea else ""
                if textarea and textarea.get("id"):
                    linked_textarea_ids.add(textarea["id"])
                
                if label in raw_target:
                    # Ghép chuỗi thô
                    if raw_target[label]:
                        raw_target[label] += " " + text
                    else:
                        raw_target[label] = text
                        
                    # Chuẩn hóa
                    norm_val = normalize_field(label, text)
                    if norm_val:
                        if target[label]:
                            target[label] += " " + norm_val
                        else:
                            target[label] = norm_val
                            
                    # Thêm boxes
                    field_boxes[label].append(bbox)
                    field_polygons[label].append(poly)
            
            # Xử lý các textarea độc lập (json_only)
            for ta in textareas:
                if ta["id"] not in linked_textarea_ids:
                    field_name = ta["from_name"]
                    if field_name in raw_target:
                        text = ta["text"]
                        if raw_target[field_name]:
                            raw_target[field_name] += " " + text
                        else:
                            raw_target[field_name] = text
                            
                        norm_val = normalize_field(field_name, text)
                        if norm_val:
                            if target[field_name]:
                                target[field_name] += " " + norm_val
                            else:
                                target[field_name] = norm_val
                    
            # Chuẩn hóa lại target sau khi ghép
            for field in ["date", "total"]:
                if target[field]:
                    target[field] = normalize_field(field, target[field])
            
            # Xác định annotation_level
            has_boxes = any(len(v) > 0 for v in field_boxes.values())
            annotation_level = "json_and_boxes" if has_boxes else "json_only"
            
            stem_id = f"self_{Path(filename).stem}"
            sample = {
                "id": stem_id,
                "group_id": stem_id,
                "store_group": normalize_store_name(raw_target["store_name"]) or "self_unknown",
                "source": "self_collected",
                "image_path": os.path.relpath(img_path, project_root).replace("\\", "/"),
                "width": width,
                "height": height,
                "annotation_level": annotation_level,
                "raw_target": raw_target,
                "target": target,
                "field_boxes": field_boxes,
                "field_polygons": field_polygons,
                "ocr_cache_path": f"data/interim/ocr_cache/{stem_id}.json"
            }
            
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Đã chuyển đổi thành công {count} mẫu từ Label Studio sang {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi Label Studio export JSON sang unified JSONL")
    parser.add_argument("--json_path", type=str, default="data/raw/self_collected/label_studio.json", help="Đường dẫn file JSON xuất ra từ LS")
    parser.add_argument("--images_dir", type=str, default="data/raw/self_collected/images", help="Thư mục chứa ảnh")
    parser.add_argument("--output_jsonl", type=str, default="data/interim/self_unified.jsonl", help="File JSONL đầu ra")
    parser.add_argument("--project_root", type=str, default=".", help="Gốc project để tính đường dẫn tương đối")
    args = parser.parse_args()
    
    convert_labelstudio(args.json_path, args.images_dir, args.output_jsonl, args.project_root)
