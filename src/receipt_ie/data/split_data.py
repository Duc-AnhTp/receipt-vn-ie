import json
import argparse
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
import sys

# Thêm src vào path để import
sys.path.append(str(Path(__file__).parent.parent.parent))

def get_image_md5(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def write_jsonl(items: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def check_leakage(train: list, val: list, test: list) -> bool:
    """
    Kiểm tra xem có sự rò rỉ dữ liệu (leakage) giữa các tập hay không bằng MD5 hash.
    """
    train_hashes = {get_image_md5(x["image_path"]) for x in train if get_image_md5(x["image_path"])}
    val_hashes = {get_image_md5(x["image_path"]) for x in val if get_image_md5(x["image_path"])}
    test_hashes = {get_image_md5(x["image_path"]) for x in test if get_image_md5(x["image_path"])}
    
    # Giao nhau của các tập
    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)
    
    has_leak = False
    if leak_train_val:
        print(f"[CẢNH BÁO] Rò rỉ giữa Train và Val! Số lượng ảnh trùng: {len(leak_train_val)}")
        has_leak = True
    if leak_train_test:
        print(f"[CẢNH BÁO] Rò rỉ giữa Train và Test! Số lượng ảnh trùng: {len(leak_train_test)}")
        has_leak = True
    if leak_val_test:
        print(f"[CẢNH BÁO] Rò rỉ giữa Val và Test! Số lượng ảnh trùng: {len(leak_val_test)}")
        has_leak = True
        
    if not has_leak:
        print("[OK] Không phát hiện bất kỳ sự rò rỉ dữ liệu nào giữa các tập split!")
        
    return not has_leak

def split_main(items: list, out_dir: Path) -> tuple:
    """
    Main Split Protocol:
    - 70/15/15 train/val/test
    - Group split theo group_id
    - Stratify theo source (mc_ocr_2021, self_collected)
    """
    print("\n--- Bắt đầu chia Main Split Protocol ---")
    
    # Gom nhóm theo group_id
    group_items = defaultdict(list)
    for item in items:
        group_items[item["group_id"]].append(item)
        
    unique_groups = list(group_items.keys())
    
    # Xác định class đại diện cho từng group để stratify (chọn source phổ biến nhất trong group)
    group_sources = []
    for g in unique_groups:
        srcs = [x["source"] for x in group_items[g]]
        # Lấy source xuất hiện nhiều nhất
        most_common_src = max(set(srcs), key=srcs.count)
        group_sources.append(most_common_src)
        
    # Split group: 70% train, 30% temp
    train_groups, temp_groups, y_train, y_temp = train_test_split(
        unique_groups, group_sources,
        test_size=0.30,
        random_state=42,
        stratify=group_sources
    )
    
    # Split temp thành val (15%) và test (15%)
    val_groups, test_groups, _, _ = train_test_split(
        temp_groups, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
    
    train_items = []
    for g in train_groups:
        train_items.extend(group_items[g])
        
    val_items = []
    for g in val_groups:
        val_items.extend(group_items[g])
        
    test_items = []
    for g in test_groups:
        test_items.extend(group_items[g])
        
    print(f"Kết quả Main Split:")
    print(f"  - Train: {len(train_items)} mẫu")
    print(f"  - Val:   {len(val_items)} mẫu")
    print(f"  - Test:  {len(test_items)} mẫu")
    
    # Lưu ra đĩa
    write_jsonl(train_items, out_dir / "train.jsonl")
    write_jsonl(val_items, out_dir / "val.jsonl")
    write_jsonl(test_items, out_dir / "test.jsonl")
    
    # Kiểm tra leak
    check_leakage(train_items, val_items, test_items)
    
    return train_items, val_items, test_items

def split_stress(items: list, out_dir: Path):
    """
    Stress-Test Split Protocol (Store Held-out):
    - Đưa một số store_group lớn hoàn toàn vào tập test và val, không xuất hiện ở train.
    """
    print("\n--- Bắt đầu chia Stress-Test Split Protocol (Store Held-out) ---")
    
    # Gom nhóm theo store_group
    store_groups = defaultdict(list)
    for item in items:
        store_groups[item["store_group"]].append(item)
        
    unique_stores = list(store_groups.keys())
    
    # Lọc bỏ các store mang nhãn unknown ra khỏi danh sách held-out đặc trưng
    known_stores = [s for s in unique_stores if s not in ["mcocr_unknown", "self_unknown"]]
    
    # Sắp xếp các store theo số lượng ảnh giảm dần
    known_stores = sorted(known_stores, key=lambda s: len(store_groups[s]), reverse=True)
    
    # Chọn ra khoảng 20% số store lớn để làm tập test và val stress-test
    # Ví dụ: 10% test_stores, 10% val_stores
    num_heldout = max(2, int(len(known_stores) * 0.20))
    test_stores = known_stores[:num_heldout//2]
    val_stores = known_stores[num_heldout//2:num_heldout]
    train_stores = known_stores[num_heldout:] + ["mcocr_unknown", "self_unknown"]
    
    train_items = []
    for s in train_stores:
        train_items.extend(store_groups[s])
        
    val_items = []
    for s in val_stores:
        val_items.extend(store_groups[s])
        
    test_items = []
    for s in test_stores:
        test_items.extend(store_groups[s])
        
    print(f"Kết quả Stress-Test (Store Held-out) Split:")
    print(f"  - Train (Stores thông thường): {len(train_items)} mẫu")
    print(f"  - Val (Stores bị held-out):    {len(val_items)} mẫu")
    print(f"  - Test (Stores bị held-out):   {len(test_items)} mẫu")
    
    # Lưu ra đĩa
    write_jsonl(train_items, out_dir / "train_stress.jsonl")
    write_jsonl(val_items, out_dir / "val_stress.jsonl")
    write_jsonl(test_items, out_dir / "test_stress.jsonl")
    
    # Kiểm tra leak
    check_leakage(train_items, val_items, test_items)

def run_split(mcocr_jsonl: str, self_jsonl: str, out_dir_str: str):
    mcocr_path = Path(mcocr_jsonl)
    self_path = Path(self_jsonl)
    out_dir = Path(out_dir_str)
    
    items = []
    # Đọc MC-OCR
    if mcocr_path.exists():
        print(f"Đang đọc {mcocr_path}...")
        items.extend(read_jsonl(mcocr_path))
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {mcocr_jsonl}")
        
    # Đọc tự thu thập
    if self_path.exists():
        print(f"Đang đọc {self_path}...")
        items.extend(read_jsonl(self_path))
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {self_jsonl}")
        
    if not items:
        print("Không có dữ liệu để thực hiện chia split. Bỏ qua.")
        return
        
    total_raw = len(items)
    
    # Loại bỏ trùng lặp ảnh bằng MD5 hash
    print("\nĐang loại bỏ trùng lặp ảnh bằng MD5 hash...")
    unique_items = []
    seen_hashes = set()
    dup_count = 0
    for item in items:
        h = get_image_md5(item["image_path"])
        if not h:
            unique_items.append(item)
            continue
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_items.append(item)
        else:
            dup_count += 1
            
    print(f"Đã loại bỏ {dup_count} ảnh trùng lặp. Số ảnh độc nhất: {len(unique_items)} / {total_raw}")
    
    # Chia Main Split
    split_main(unique_items, out_dir)
    
    # Chia Stress-test Split
    split_stress(unique_items, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chia train/val/test dữ liệu biên lai tiếng Việt")
    parser.add_argument("--mcocr_jsonl", type=str, default="data/interim/mcocr_unified.jsonl", help="Đường dẫn MC-OCR JSONL")
    parser.add_argument("--self_jsonl", type=str, default="data/interim/self_unified.jsonl", help="Đường dẫn tự thu thập JSONL")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Thư mục ghi nhận kết quả splits")
    args = parser.parse_args()
    
    run_split(args.mcocr_jsonl, args.self_jsonl, args.out_dir)
