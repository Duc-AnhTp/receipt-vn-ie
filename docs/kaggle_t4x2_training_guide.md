# Kaggle T4x2 Training Guide

Huong dan nay mo ta luong chuan de clone code tu GitHub, lay du lieu/checkpoint tu Kaggle Dataset, validate, build OCR cache, train/evaluate tren GPU T4x2 va tai artifact ve.

## 1. Kaggle Setup

GitHub chi chua code, docs, configs, requirements, scripts va tests. Khong push `data/`, `checkpoints/`, `outputs/`, `*.zip`.

Tao 2 Kaggle Dataset:

- `receipt-vn-ie-data`: chua `receipt_dataset.zip`, giai nen ra cau truc `data/...`.
- `receipt-vn-ie-checkpoints`: optional, dung cho demo/evaluate/resume. Layout chuan:
  - `donut/receipt_ie/finetune/best_model`
  - `layoutxlm/receipt_ie/ocr_cache/best_model`
  - hoac co prefix `checkpoints/...` tuong ung.

Kaggle Notebook:

- Accelerator: `GPU T4 x2`
- Internet: bat neu can install dependencies/model pretrained
- Add Data: `receipt-vn-ie-data`, optional `receipt-vn-ie-checkpoints`

Bootstrap:

```bash
git clone <github-url> /kaggle/working/receipt-vn-ie
cd /kaggle/working/receipt-vn-ie
bash scripts/kaggle_bootstrap.sh
```

Kiem tra GPU:

```bash
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

## 2. Data + OCR

Validate:

```bash
bash scripts/01_validate_data.sh
```

Dieu kien pass: `total_errors = 0`, date dung `YYYY-MM-DD`, total chi gom digits, image/box khong loi nghiem trong.

OCR preprocess ablation:

```bash
bash scripts/02b_ablate_ocr_preprocess.sh
```

Chon profile tot nhat theo rough OCR quality + latency. Mac dinh khuyen nghi:

```bash
export PREPROCESS_PROFILE=resize
```

Build OCR cache full tu processed split:

```bash
PREPROCESS_PROFILE=resize bash scripts/02_build_ocr_cache.sh
```

Smoke OCR cache:

```bash
python -m receipt_ie.ocr.build_ocr_cache \
  --data_files data/processed/train.jsonl \
  --limit 20 \
  --preprocess_profile resize \
  --overwrite
```

## 3. LayoutXLM First

Visualize BIO labels truoc khi train:

```bash
python -m receipt_ie.visualization.visualize_layoutxlm_labels \
  --jsonl data/processed/train.jsonl \
  --limit 50 \
  --output_dir outputs/debug/layoutxlm_labels \
  --overlap_threshold 0.4
```

Train LayoutXLM tren T4x2:

```bash
USE_ACCELERATE=1 NUM_PROCESSES=2 bash scripts/03_train_layoutxlm.sh
```

Fallback 1 GPU/no accelerate:

```bash
USE_ACCELERATE=0 bash scripts/03_train_layoutxlm.sh
```

Checkpoint chuan cho app/evaluate:

```text
checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model
```

## 4. Evaluate + Error Analysis

Evaluate Baseline + LayoutXLM truoc, bo Donut neu chua train:

```bash
RUN_DONUT=0 bash scripts/05_evaluate_all.sh
```

Evaluate du 3 model:

```bash
RUN_BASELINE=1 RUN_LAYOUTXLM=1 RUN_DONUT=1 bash scripts/05_evaluate_all.sh
```

`scripts/05_evaluate_all.sh` se xuat:

- `outputs/metrics/main_metrics.json`
- `outputs/metrics/latency_by_method.json`
- `outputs/error_analysis/error_by_field.csv`

## 5. Donut After Pipeline Is Clean

Train Donut sau khi validate/OCR/LayoutXLM da on. Khuyen nghi su dung che do `full` (warmup tren CORD v2 truoc khi finetune) de dat do chinh xac cao nhat:

```bash
# 1. Download & convert CORD v2 (chi can chay 1 lan truoc khi train)
python -m receipt_ie.data.download_cord
python -m receipt_ie.data.convert_cord

# 2. Train Donut o che do full (tu dong warmup CORD -> finetune Tieng Viet)
USE_ACCELERATE=1 NUM_PROCESSES=2 bash scripts/04_train_donut.sh --mode full
```

Fallback hoac chi chay finetune (neu da co checkpoint warmup):

```bash
USE_ACCELERATE=1 NUM_PROCESSES=2 bash scripts/04_train_donut.sh --mode finetune
```

Checkpoint chuan:

```text
checkpoints/donut/receipt_ie/finetune/best_model
```

## 6. Smoke Test

Chay smoke tren Kaggle:

```bash
bash scripts/07_smoke_kaggle.sh
```

Thu tu go/no-go truoc train full:

1. `pytest` pass.
2. Validate train/val/test pass.
3. OCR cache build full pass.
4. OCR quality report khong qua te.
5. BIO visualization nhin hop ly.
6. Checkpoint path dung `best_model`.
7. Accelerate T4x2 co fallback ro rang.

## 7. Artifacts Can Tai Ve

- `outputs/metrics/`
- `outputs/error_analysis/`
- `outputs/ocr/ocr_quality_sample.csv`
- `outputs/predictions/`
- `checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model/`
- `checkpoints/donut/receipt_ie/finetune/best_model/`
