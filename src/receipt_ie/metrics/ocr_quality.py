import json
import argparse
import os
import re
from pathlib import Path
import pandas as pd
from rapidfuzz.fuzz import partial_ratio

from receipt_ie.postprocess.total_extractor import strip_accents

def normalize_for_compare(s: str) -> str:
    s = strip_accents(s or "").lower()
    return re.sub(r"[^\w\s]", "", s).strip()

def main():
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng OCR thô so với Ground Truth.")
    parser.add_argument("--jsonl", required=True, help="Đường dẫn file JSONL dataset (ví dụ: val.jsonl)")
    parser.add_argument("--ocr_cache_dir", default="data/interim/ocr_cache", help="Thư mục chứa OCR cache JSON")
    parser.add_argument("--output_csv", default="outputs/ocr_quality_sample.csv", help="Đường dẫn lưu file CSV báo cáo")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số lượng mẫu đánh giá")
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Không tìm thấy file dataset tại {jsonl_path}")
        return
        
    ocr_cache_dir = Path(args.ocr_cache_dir)
    
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    if args.limit is not None:
        samples = samples[:args.limit]
        
    fields = ["store_name", "date", "total", "address"]
    quality_records = []
    
    print(f"Bắt đầu đánh giá chất lượng OCR trên {len(samples)} mẫu từ {jsonl_path.name}...")
    
    for sample in samples:
        sample_id = sample.get("id")
        image_path = sample.get("image_path")
        target = sample.get("target") or {}
        
        # Tìm file ocr cache
        cache_path = sample.get("ocr_cache_path")
        candidates = []
        if cache_path:
            candidates.append(Path(cache_path))
            candidates.append(ocr_cache_dir / Path(cache_path).name)
        candidates.append(ocr_cache_dir / f"{sample_id}.json")
        
        ocr_data = None
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        ocr_data = json.load(f)
                    break
                except Exception:
                    pass
                    
        if not ocr_data:
            # Không có ocr cache cho sample này, gán độ phủ = 0
            record = {
                "id": sample_id,
                "image_path": image_path,
                "has_ocr_cache": False,
                "store_name_match": 0.0,
                "date_match": 0.0,
                "total_match": 0.0,
                "address_match": 0.0,
                "avg_match": 0.0
            }
            quality_records.append(record)
            continue
            
        # Ghép text OCR phẳng
        ocr_words = ocr_data.get("words", [])
        ocr_text = " ".join([w.get("text", "") for w in ocr_words if w.get("text")]).strip()
        ocr_text_clean = normalize_for_compare(ocr_text)
        
        scores = {}
        for f in fields:
            gold_val = target.get(f, "")
            gold_clean = normalize_for_compare(gold_val)
            
            if not gold_clean:
                # Nếu Ground Truth trống, coi như OCR khớp hoàn hảo 100% trường đó
                scores[f] = 1.0
            else:
                if not ocr_text_clean:
                    scores[f] = 0.0
                else:
                    # Tính tỉ lệ trùng khớp chuỗi con mờ (partial fuzzy match ratio)
                    # Trả về giá trị trong dải [0.0, 1.0]
                    scores[f] = partial_ratio(gold_clean, ocr_text_clean) / 100.0
                    
        avg_score = sum(scores.values()) / len(fields)
        
        record = {
            "id": sample_id,
            "image_path": image_path,
            "has_ocr_cache": True,
            "store_name_match": round(scores["store_name"], 4),
            "date_match": round(scores["date"], 4),
            "total_match": round(scores["total"], 4),
            "address_match": round(scores["address"], 4),
            "avg_match": round(avg_score, 4)
        }
        quality_records.append(record)
        
    df = pd.DataFrame(quality_records)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    
    # Tính toán kết quả trung bình tổng thể
    total_samples = len(df)
    cached_samples = df["has_ocr_cache"].sum()
    
    print(f"\n=== BÁO CÁO CHẤT LƯỢNG OCR TỔNG THỂ ===")
    print(f"Tổng số mẫu đánh giá: {total_samples}")
    print(f"Số mẫu có OCR cache: {cached_samples} ({cached_samples/total_samples*100:.2f}%)")
    
    if cached_samples > 0:
        df_cached = df[df["has_ocr_cache"]]
        print(f"Điểm rough match trung bình từng trường:")
        print(f"  - store_name: {df_cached['store_name_match'].mean() * 100:.2f}%")
        print(f"  - date:       {df_cached['date_match'].mean() * 100:.2f}%")
        print(f"  - total:      {df_cached['total_match'].mean() * 100:.2f}%")
        print(f"  - address:    {df_cached['address_match'].mean() * 100:.2f}%")
        print(f"  - Trung bình:  {df_cached['avg_match'].mean() * 100:.2f}%")
        
    print(f"\nBáo cáo chi tiết chất lượng OCR lưu tại: {out_path}")

if __name__ == "__main__":
    main()
