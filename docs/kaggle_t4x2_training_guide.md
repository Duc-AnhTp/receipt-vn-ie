# Kaggle T4x2 Training Guide

Hướng dẫn này mô tả luồng chuẩn để clone code từ GitHub, lấy dữ liệu từ Kaggle Dataset, train/evaluate trên GPU T4x2 và tải artifact về.

## 1. Chuẩn bị trước khi lên Kaggle

### GitHub repo

Chỉ push code và metadata:

- `src/`
- `configs/`
- `docs/`
- `requirements/`
- `scripts/`
- `tests/`
- `README.md`
- `pyproject.toml`

Không push:

- `data/`
- `checkpoints/`
- `outputs/`
- `*.zip`

`.gitignore` hiện đã loại các thư mục/file lớn này.

### Kaggle Dataset

Tạo 2 Kaggle Dataset riêng:

1. `receipt-vn-ie-data`
   - Chứa `receipt_dataset.zip`
   - Zip sau khi giải nén phải tạo được cấu trúc `data/...`, ví dụ:
     - `data/processed/train.jsonl`
     - `data/processed/val.jsonl`
     - `data/processed/test.jsonl`
     - `data/interim/ocr_cache/`
     - `data/raw/...`

2. `receipt-vn-ie-checkpoints`
   - Optional, dùng để demo/evaluate/resume.
   - Layout được hỗ trợ:
     - `donut/receipt_ie/final`
     - `layoutxlm/receipt_ie/final`
   - Hoặc:
     - `checkpoints/donut/receipt_ie/final`
     - `checkpoints/layoutxlm/receipt_ie/final`

## 2. Tạo Kaggle Notebook

Trong Kaggle Notebook:

- Accelerator: `GPU T4 x2`
- Internet: bật nếu cần `pip install`, tải model pretrained hoặc dependency.
- Add Data:
  - `receipt-vn-ie-data`
  - `receipt-vn-ie-checkpoints` nếu có checkpoint sẵn

Cell bootstrap:

```bash
git clone <github-url> /kaggle/working/receipt-vn-ie
cd /kaggle/working/receipt-vn-ie
bash scripts/kaggle_bootstrap.sh
```

Bootstrap sẽ:

- cài package project
- cài app/OCR dependencies
- thử cài `paddlepaddle-gpu`
- giải nén `receipt_dataset.zip`
- copy checkpoint về `checkpoints/.../final`
- in thông tin CUDA/GPU

Nếu `paddlepaddle-gpu` không tương thích môi trường Kaggle hiện tại, ưu tiên dùng OCR cache đã build sẵn trong dataset. Chỉ build lại OCR khi PaddleOCR hoạt động ổn.

## 3. Kiểm tra dữ liệu

```bash
cd /kaggle/working/receipt-vn-ie
bash scripts/01_validate_data.sh
```

Kiểm tra các file:

- `outputs/metrics/train_validation_report.json`
- `outputs/metrics/val_validation_report.json`
- `outputs/metrics/test_validation_report.json`
- `outputs/metrics/field_missing_rate.csv`

Điều kiện pass:

- `total_errors = 0`
- `date` đúng `YYYY-MM-DD`
- `total` chỉ gồm chữ số
- `image_path` tồn tại
- `field_boxes` không sai tọa độ nghiêm trọng

## 4. Kiểm tra hoặc build OCR cache

Nếu dataset đã có `data/interim/ocr_cache`, kiểm tra chất lượng OCR mẫu:

```bash
python -m receipt_ie.metrics.ocr_quality \
  --jsonl_path data/processed/train.jsonl \
  --ocr_cache_dir data/interim/ocr_cache \
  --output_csv outputs/ocr/ocr_quality_sample.csv \
  --limit 30
```

Nếu cần build lại OCR cache:

```bash
bash scripts/02_build_ocr_cache.sh
```

Lưu ý: OCR cache là trung tâm của Baseline và LayoutXLM. Nếu OCR cache kém, train LayoutXLM sẽ học trên input nhiễu.

## 5. Train LayoutXLM trước

LayoutXLM là model chính cho `store_name` và `address`.

```bash
bash scripts/03_train_layoutxlm.sh
```

Kaggle T4x2 multi-GPU:

```bash
NUM_GPUS=2 bash scripts/03_train_layoutxlm.sh
```

Mặc định:

- mode: `ocr_cache`
- config: `configs/layoutxlm.yaml`
- output: `checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model`
- final checkpoint: `checkpoints/layoutxlm/receipt_ie/final` (auto-synced by the training script)

Sau khi train xong, copy checkpoint tốt nhất về đường dẫn chuẩn để evaluate/demo:

```bash
mkdir -p checkpoints/layoutxlm/receipt_ie/final
cp -R checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model/. checkpoints/layoutxlm/receipt_ie/final/
```

Manual copy is only needed for old checkpoints. New training runs sync this path automatically on rank 0.

Nếu muốn phân tích OCR upper bound:

```bash
LAYOUTXLM_MODE=oracle_ocr bash scripts/03_train_layoutxlm.sh
```

## 6. Train Donut sau

Donut dùng để so sánh OCR-free. Luồng chính không dùng CORD.

```bash
bash scripts/04_train_donut.sh
```

Kaggle T4x2 multi-GPU:

```bash
NUM_GPUS=2 bash scripts/04_train_donut.sh
```

Script này chạy:

```bash
python -m receipt_ie.training.train_donut --mode finetune
```

Sau khi train xong, copy checkpoint tốt nhất về đường dẫn chuẩn:

```bash
mkdir -p checkpoints/donut/receipt_ie/final
cp -R checkpoints/donut/receipt_ie/finetune/best_model/. checkpoints/donut/receipt_ie/final/
```

Manual copy is only needed for old checkpoints. New training runs sync this path automatically on rank 0.

## 7. Evaluate

Evaluate Baseline + LayoutXLM trước, bỏ qua Donut nếu chưa train:

```bash
RUN_DONUT=0 bash scripts/05_evaluate_all.sh
```

Evaluate đủ 3 model khi đã có checkpoint Donut:

```bash
RUN_BASELINE=1 RUN_LAYOUTXLM=1 RUN_DONUT=1 bash scripts/05_evaluate_all.sh
```

Có thể giới hạn số mẫu để smoke test:

```bash
LIMIT=3 RUN_DONUT=0 bash scripts/05_evaluate_all.sh
```

Output chính:

- `outputs/predictions/baseline_test.jsonl`
- `outputs/predictions/layoutxlm_test.jsonl`
- `outputs/predictions/donut_test.jsonl`
- `outputs/metrics/main_metrics.json`
- `outputs/metrics/latency_by_method.json`
- `outputs/metrics/*_metrics.json`

## 8. Error Analysis

```bash
bash scripts/06_error_analysis.sh
```

Output:

- `outputs/error_analysis/error_by_field.csv`

Các nhóm lỗi:

- `EMPTY_PRED`
- `FORMAT_ERROR`
- `OCR_MISS`
- `OCR_WRONG`
- `POSTPROCESS_BAD`
- `MODEL_BAD`
- `LABEL_BAD`

Donut là OCR-free nên không được gán `OCR_MISS` hoặc `OCR_WRONG`.

## 9. Artifact cần tải về

Tải các thư mục/file sau từ Kaggle Output:

- `outputs/metrics/`
- `outputs/error_analysis/`
- `outputs/ocr/ocr_quality_sample.csv`
- `outputs/predictions/`
- `checkpoints/layoutxlm/receipt_ie/final/`
- `checkpoints/donut/receipt_ie/final/`

Nếu checkpoint lớn, nên tạo Kaggle Dataset version mới thay vì tải thủ công.

## 10. Thứ tự khuyến nghị

1. Bootstrap
2. Validate data
3. Kiểm tra OCR quality
4. Train LayoutXLM
5. Evaluate Baseline + LayoutXLM
6. Error analysis
7. Train Donut
8. Evaluate đủ 3 model
9. Tải metrics/checkpoints/artifacts
