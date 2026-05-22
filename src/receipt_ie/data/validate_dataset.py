import json
import argparse
import hashlib
from pathlib import Path
import unicodedata
from PIL import Image
import pandas as pd
from collections import defaultdict

from receipt_ie.data.schemas import FIELDS

def compute_md5(path: Path) -> str:
    """
    Tính MD5 hash của file để so khớp trùng lặp.
    """
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def is_nfc(s: str) -> bool:
    """
    Kiểm tra xem chuỗi có ở định dạng Unicode NFC hay không.
    """
    if not s:
        return True
    return s == unicodedata.normalize("NFC", s)

def validate_jsonl(jsonl_path: str, report_dir: str):
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Không tìm thấy file JSONL tại {jsonl_path} để validate.")
        return
        
    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    total_samples = len(samples)
    print(f"Bắt đầu validate {total_samples} mẫu trong {jsonl_file.name}...")
    
    errors = []
    warnings = []
    
    # MD5 mapping để phát hiện trùng lặp ảnh
    md5_map = defaultdict(list)
    id_set = set()
    
    missing_counts = {field: 0 for field in FIELDS}
    invalid_date_counts = 0
    invalid_total_counts = 0
    
    for idx, sample in enumerate(samples):
        s_id = sample.get("id")
        
        # 1. Kiểm tra ID duy nhất
        if not s_id:
            errors.append(f"Mẫu số {idx} thiếu trường 'id'")
        elif s_id in id_set:
            errors.append(f"Trùng lặp 'id': {s_id}")
        else:
            id_set.add(s_id)
            
        # 2. Kiểm tra file ảnh tồn tại và đọc được
        img_path_str = sample.get("image_path")
        if not img_path_str:
            errors.append(f"Mẫu {s_id}: Thiếu 'image_path'")
        else:
            img_path = Path(img_path_str)
            if not img_path.exists():
                errors.append(f"Mẫu {s_id}: Không tìm thấy file ảnh tại {img_path_str}")
            else:
                try:
                    # Thử mở ảnh bằng PIL
                    with Image.open(img_path) as img:
                        img.verify()
                        
                    # Tính MD5 và lưu trữ
                    img_md5 = compute_md5(img_path)
                    md5_map[img_md5].append(s_id)
                except Exception as e:
                    errors.append(f"Mẫu {s_id}: Lỗi đọc ảnh {img_path_str} - {str(e)}")
                    
        # 3. Kiểm tra targets schema
        target = sample.get("target")
        raw_target = sample.get("raw_target")
        if target is None:
            errors.append(f"Mẫu {s_id}: Thiếu trường 'target'")
        else:
            # Check đủ 4 keys
            for field in FIELDS:
                if field not in target:
                    errors.append(f"Mẫu {s_id}: 'target' thiếu key '{field}'")
                else:
                    val = target[field]
                    if not val:
                        missing_counts[field] += 1
                    else:
                        # Check Unicode NFC
                        if not is_nfc(val):
                            warnings.append(f"Mẫu {s_id}: Giá trị field '{field}' chưa được chuẩn hóa NFC: '{val}'")
                            
                        # Check format ngày
                        if field == "date" and val:
                            import re
                            if not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
                                errors.append(f"Mẫu {s_id}: Định dạng ngày không hợp lệ '{val}', phải là YYYY-MM-DD")
                                invalid_date_counts += 1
                                
                        # Check format tiền
                        if field == "total" and val:
                            if not val.isdigit():
                                errors.append(f"Mẫu {s_id}: Định dạng số tiền không hợp lệ '{val}', chỉ được chứa chữ số")
                                invalid_total_counts += 1

        # Check raw_target
        if raw_target is None:
            warnings.append(f"Mẫu {s_id}: Thiếu 'raw_target'")

        # 4. Kiểm tra bounding boxes
        field_boxes = sample.get("field_boxes", {})
        for field, boxes in field_boxes.items():
            for box_idx, box in enumerate(boxes):
                if len(box) != 4:
                    errors.append(f"Mẫu {s_id}: box {box_idx} của {field} không đủ 4 phần tử [x0, y0, x1, y1]")
                else:
                    x0, y0, x1, y1 = box
                    if x0 >= x1 or y0 >= y1:
                        errors.append(f"Mẫu {s_id}: box {box_idx} của {field} bị lỗi tọa độ: x0={x0}, x1={x1}, y0={y0}, y1={y1}")

    # 5. Phát hiện trùng lặp MD5
    duplicate_images = 0
    for md5_val, ids in md5_map.items():
        if len(ids) > 1:
            warnings.append(f"Phát hiện ảnh trùng lặp (cùng MD5 hash): {ids}")
            duplicate_images += len(ids) - 1
            
    # Ghi nhận tỷ lệ missing rates
    missing_rates = {field: missing_counts[field] / max(total_samples, 1) for field in FIELDS}
    
    # Output file báo cáo CSV cho missing rates
    missing_df = pd.DataFrame({
        "Field": FIELDS,
        "Missing_Count": [missing_counts[f] for f in FIELDS],
        "Missing_Rate": [missing_rates[f] for f in FIELDS]
    })
    missing_df.to_csv(rep_dir / "field_missing_rate.csv", index=False)
    
    # Tạo báo cáo JSON
    report = {
        "dataset_name": jsonl_file.name,
        "total_samples": total_samples,
        "total_errors": len(errors),
        "total_warnings": len(warnings),
        "duplicate_images_found": duplicate_images,
        "field_missing_counts": missing_counts,
        "field_missing_rates": missing_rates,
        "invalid_date_count": invalid_date_counts,
        "invalid_total_count": invalid_total_counts,
        "errors": errors[:100],  # Chỉ lưu tối đa 100 lỗi đầu tiên
        "warnings": warnings[:100]
    }
    
    report_file = rep_dir / f"{jsonl_file.stem}_validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        
    print(f"Đã hoàn thành validation cho {jsonl_file.name}:")
    print(f"  - Số lỗi: {len(errors)}")
    print(f"  - Số cảnh báo: {len(warnings)}")
    print(f"  - Số ảnh trùng lặp: {duplicate_images}")
    print(f"Báo cáo chi tiết đã lưu tại {report_file}")
    
    return len(errors) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xác thực tính toàn vẹn của tập dữ liệu JSONL")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Đường dẫn file JSONL cần validate")
    parser.add_argument("--report_dir", type=str, default="outputs/metrics", help="Thư mục ghi nhận báo cáo")
    args = parser.parse_args()
    
    validate_jsonl(args.jsonl_path, args.report_dir)
