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

### CORD v2 (Warm-up)
```bash
python -m receipt_ie.data.download_cord
```

## Pipeline chạy thử

```bash
# 1. Convert dữ liệu sang unified JSONL
python -m receipt_ie.data.convert_mcocr
python -m receipt_ie.data.convert_labelstudio
python -m receipt_ie.data.convert_cord

# 2. Chia train/val/test
python -m receipt_ie.data.split_data

# 3. Validate dữ liệu
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/train.jsonl
```

## Đánh giá

| Phương pháp | Store EM | Date EM | Total EM | Address EM | Macro NES |
|---|---|---|---|---|---|
| Baseline (Rule-based) | — | — | — | — | — |
| Donut | — | — | — | — | — |
| VietOCR + LayoutXLM | — | — | — | — | — |

*Kết quả sẽ được cập nhật sau khi huấn luyện xong.*

## Tests

```bash
pytest tests/
```

## License

MIT
