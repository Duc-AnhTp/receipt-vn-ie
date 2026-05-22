import argparse
from datasets import load_dataset
from pathlib import Path

def download_cord(output_dir: str):
    """
    Tải tập dữ liệu CORD v2 từ Hugging Face và lưu về thư mục raw local.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Đang tải tập dữ liệu naver-clova-ix/cord-v2 từ Hugging Face...")
    dataset = load_dataset("naver-clova-ix/cord-v2")
    
    print(f"Đang lưu dữ liệu về {out_path}...")
    dataset.save_to_disk(str(out_path))
    print("Hoàn tất tải CORD v2!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải dữ liệu CORD v2 từ Hugging Face")
    parser.add_argument("--output_dir", type=str, default="data/raw/cord_v2", help="Thư mục lưu dữ liệu")
    args = parser.parse_args()
    
    download_cord(args.output_dir)
