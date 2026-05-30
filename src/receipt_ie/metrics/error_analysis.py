import json
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from rapidfuzz.distance import Levenshtein

from receipt_ie.postprocess.total_extractor import strip_accents

def normalize_for_compare(s: str) -> str:
    s = strip_accents(s or "").lower()
    return re.sub(r"[^\w\s]", "", s).strip()

def classify_error(
    field: str,
    gold_val: str,
    pred_val: str,
    raw_pred_val: str,
    ocr_words: List[Dict[str, Any]]
) -> str:
    """
    Phân loại lỗi của một trường cụ thể thành 7 nhóm lỗi cố định.
    """
    gold_val = str(gold_val or "").strip()
    pred_val = str(pred_val or "").strip()
    raw_pred_val = str(raw_pred_val or "").strip()
    
    if gold_val == pred_val:
        return "NONE"
        
    # 1. EMPTY_PRED: Ground truth có giá trị nhưng prediction rỗng
    if gold_val and not pred_val:
        return "EMPTY_PRED"
        
    # 2. LABEL_BAD: GT thiếu nhưng pred có trích xuất được (hoặc GT bị sai hiển nhiên)
    if not gold_val and pred_val:
        return "LABEL_BAD"
        
    # 3. FORMAT_ERROR: Định dạng prediction sai
    if field == "date" and pred_val:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", pred_val):
            return "FORMAT_ERROR"
    elif field == "total" and pred_val:
        if not pred_val.isdigit():
            return "FORMAT_ERROR"
            
    # Lấy text OCR phẳng
    ocr_text = " ".join([w.get("text", "") for w in ocr_words if w.get("text")]).strip()
    ocr_text_clean = normalize_for_compare(ocr_text)
    gold_clean = normalize_for_compare(gold_val)
    
    # 4. OCR_MISS: Text gold hoàn toàn không xuất hiện trong kết quả OCR
    if gold_clean and gold_clean not in ocr_text_clean:
        # Kiểm tra thêm độ tương đồng Levenshtein để tránh bỏ sót do lỗi chính tả nhỏ
        # Nếu khoảng cách chỉnh sửa quá lớn trên toàn bộ ocr_text
        return "OCR_MISS"
        
    # 5. POSTPROCESS_BAD: Raw prediction thô đúng/gần đúng nhưng qua normalize bị hỏng
    raw_clean = normalize_for_compare(raw_pred_val)
    if raw_clean and (gold_clean in raw_clean or Levenshtein.normalized_similarity(gold_clean, raw_clean) > 0.8):
        # Text thô gần đúng nhưng sau khi normalized bị khác
        if pred_val != gold_val:
            return "POSTPROCESS_BAD"
            
    # 6. OCR_WRONG: OCR nhận dạng sai từ khóa/ký tự làm mất/sai lệch thông tin
    # Ví dụ: gold text chứa "anan" nhưng ocr chỉ nhận dạng được "anar"
    if gold_clean not in ocr_text_clean:
        # Nếu có một từ trong OCR rất giống với gold_clean (similarity > 0.7)
        # chứng tỏ OCR nhận dạng sai
        for w in ocr_words:
            w_clean = normalize_for_compare(w.get("text", ""))
            if w_clean and Levenshtein.normalized_similarity(gold_clean, w_clean) > 0.7:
                return "OCR_WRONG"
                
    # 7. MODEL_BAD: Mô hình gán nhãn sai lớp (mặc dù OCR có đúng text đó)
    return "MODEL_BAD"

def main():
    parser = argparse.ArgumentParser(description="Phân tích lỗi chi tiết cho 4 trường thông tin.")
    parser.add_argument("--gold", required=True, help="Đường dẫn file JSONL ground truth")
    parser.add_argument("--pred", required=True, help="Đường dẫn file JSONL dự đoán")
    parser.add_argument("--ocr_cache_dir", default="data/interim/ocr_cache", help="Thư mục OCR cache")
    parser.add_argument("--output", default="outputs/error_analysis/error_by_field.csv", help="Đường dẫn file CSV đầu ra")
    args = parser.parse_args()
    
    # Đọc dữ liệu
    def read_jsonl(path):
        records = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "id" in item:
                        records[item["id"]] = item
        return records
        
    gold_records = read_jsonl(args.gold)
    pred_records = read_jsonl(args.pred)
    
    # Đọc OCR cache
    ocr_cache_dir = Path(args.ocr_cache_dir)
    
    fields = ["store_name", "date", "total", "address"]
    error_records = []
    
    # Đếm số lượng loại lỗi
    error_counts = {f: {err: 0 for err in ["NONE", "EMPTY_PRED", "FORMAT_ERROR", "OCR_MISS", "OCR_WRONG", "POSTPROCESS_BAD", "MODEL_BAD", "LABEL_BAD"]} for f in fields}
    
    for sample_id, gold_item in gold_records.items():
        pred_item = pred_records.get(sample_id)
        if not pred_item:
            continue
            
        if pred_item.get("status") == "error":
            continue
            
        gold_target = gold_item.get("target") or gold_item
        pred_normalized = pred_item.get("normalized_prediction") or pred_item
        pred_raw = pred_item.get("prediction") or {}
        
        # Load OCR words cho sample này
        ocr_words = []
        cache_path = gold_item.get("ocr_cache_path")
        candidates = []
        if cache_path:
            candidates.append(Path(cache_path))
            candidates.append(ocr_cache_dir / Path(cache_path).name)
        candidates.append(ocr_cache_dir / f"{sample_id}.json")
        
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        ocr_words = json.load(f).get("words", [])
                    break
                except Exception:
                    pass
                    
        for f in fields:
            g_val = gold_target.get(f, "")
            p_val = pred_normalized.get(f, "")
            p_raw = pred_raw.get(f, "")
            
            err_type = classify_error(f, g_val, p_val, p_raw, ocr_words)
            error_counts[f][err_type] += 1
            
            if err_type != "NONE":
                error_records.append({
                    "id": sample_id,
                    "field": f,
                    "gold": g_val,
                    "pred": p_val,
                    "raw_pred": p_raw,
                    "error_type": err_type
                })
                
    # Xuất file CSV phân tích chi tiết các mẫu bị lỗi
    df_details = pd.DataFrame(error_records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_details.to_csv(out_path, index=False, encoding="utf-8")
    
    # In ra bảng thống kê tổng kết
    print("\n=== THỐNG KÊ PHÂN LOẠI LỖI THEO FIELD ===")
    summary_data = []
    for f in fields:
        row = {"Field": f}
        row.update(error_counts[f])
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    summary_path = out_path.parent / "error_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nBáo cáo chi tiết các mẫu lỗi lưu tại: {out_path}")
    print(f"Báo cáo thống kê tổng hợp lưu tại: {summary_path}")

if __name__ == "__main__":
    main()
