# Receipt VN IE — Trích xuất thông tin biên lai tiếng Việt

So sánh hai kiến trúc **Donut** (End-to-End OCR-free) và **VietOCR + LayoutXLM** (OCR-based layout-aware) để trích xuất 4 trường thông tin từ biên lai tiếng Việt.

## Trường thông tin trích xuất

| Trường | Định dạng | Ví dụ |
|---|---|---|
| `store_name` | Chuỗi Unicode NFC | MINIMART ANAN |
| `date` | YYYY-MM-DD | 2020-08-09 |
| `total` | Chuỗi số nguyên | 115000 |
| `address` | Chuỗi Unicode NFC | Chợ Sủi Phú Thị Gia Lâm |

## Cấu trúc dự án

```
receipt-vn-ie/
├── configs/          # YAML configs cho data, OCR, model, app
├── data/             # Raw → Interim → Processed data pipeline
├── docs/             # Scope, annotation guideline, normalization rules
├── src/receipt_ie/   # Source code chính
│   ├── data/         # Convert, normalize, validate, split
│   ├── ocr/          # PaddleOCR detect + VietOCR recognize
│   ├── baseline/     # Rule-based extractor
│   ├── models/       # Donut & LayoutXLM model wrappers
│   ├── training/     # Training scripts
│   ├── inference/    # Inference pipelines
│   ├── metrics/      # EM, NES, CER, latency
│   ├── visualization/# Bounding box drawing, comparison
│   └── app/          # Gradio web demo
├── tests/            # Unit tests
├── notebooks/        # EDA, OCR quality check, error analysis
└── scripts/          # Shell scripts cho pipeline automation
```

## Cài đặt

```bash
# 1. Clone repo
git clone <repo-url>
cd receipt-vn-ie

# 2. Cài đặt dependencies cơ bản
pip install -e .

# 3. Cài đặt OCR engines
pip install -r requirements/ocr.txt
pip install -r requirements/paddle-cpu.txt   # hoặc paddle-gpu.txt

# 4. Cài đặt Gradio (cho web demo)
pip install -r requirements/app.txt
```

## Dữ liệu

### MC-OCR 2021
Đặt dữ liệu vào `data/raw/mc_ocr_2021/`:
- `mcocr_train_df.csv` — file CSV chứa annotation
- `train_images/` — thư mục chứa ảnh biên lai

### Tự thu thập
Đặt dữ liệu vào `data/raw/self_collected/`:
- `label_studio.json` — export từ Label Studio
- `images/` — thư mục chứa ảnh biên lai

### CORD v2 (Optional / Future Work)
CORD v2 chỉ được giữ như nhánh tham khảo cho thí nghiệm warm-up trong tương lai.
Luồng chính và kết quả báo cáo hiện tại **không dùng CORD v2**.

## Pipeline chạy thử

```bash
# 1. Convert dữ liệu sang unified JSONL
python -m receipt_ie.data.convert_mcocr
python -m receipt_ie.data.convert_labelstudio

# 2. Chia train/val/test
python -m receipt_ie.data.split_data

# 3. Validate dữ liệu
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/train.jsonl
```

## GitHub → Kaggle T4x2

Repo GitHub chỉ lưu code, docs, configs, requirements, tests và scripts. Không commit
`data/`, `checkpoints/`, `outputs/` hoặc các file `*.zip`.

Trên Kaggle, dùng 2 Dataset riêng:
- `receipt-vn-ie-data`: chứa `receipt_dataset.zip`
- `receipt-vn-ie-checkpoints`: chứa checkpoint Donut/LayoutXLM nếu cần demo, evaluate hoặc resume

Bootstrap trên Kaggle:

```bash
git clone <repo-url> /kaggle/working/receipt-vn-ie
cd /kaggle/working/receipt-vn-ie
bash scripts/kaggle_bootstrap.sh
```

Pipeline chuẩn:

```bash
bash scripts/01_validate_data.sh
bash scripts/02_build_ocr_cache.sh
bash scripts/03_train_layoutxlm.sh
bash scripts/04_train_donut.sh
bash scripts/05_evaluate_all.sh
bash scripts/06_error_analysis.sh
```

## Đánh giá

| Phương pháp | Store EM | Date EM | Total EM | Address EM | Macro EM | Macro NES | Macro CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Rule-based) | 25.21 | 74.79 | 54.27 | 4.70 | 39.74 | 54.00 | 57.71 |
| Donut | 3.85 | 7.69 | 7.26 | 4.70 | 5.88 | 6.94 | 105.23 |
| VietOCR + LayoutXLM | 29.06 | 60.68 | 69.66 | 14.53 | 43.48 | 61.86 | 39.64 |

Kết quả được đánh giá trên tập test 234 mẫu, split 70/15/15, seed = 42. Môi trường huấn luyện: Kaggle T4×2.

## Tests

```bash
pytest tests/
```

## License

MIT
