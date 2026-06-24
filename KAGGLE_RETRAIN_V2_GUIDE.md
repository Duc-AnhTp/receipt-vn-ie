# Hướng Dẫn Chạy Lại Thực Nghiệm Trên Kaggle (Retrain V2)

Tài liệu này hướng dẫn **step-by-step** để chạy lại toàn bộ thực nghiệm trên Kaggle,
tạo artifact mới cho báo cáo bảo vệ. Tổng thời gian GPU ước tính: **~6-8 giờ**.

## Tình trạng hiện tại

| Tài nguyên | Trạng thái |
|---|---|
| GPU Kaggle còn lại | 26 giờ |
| Checkpoint Donut `best_model` | ✅ Còn nguyên (771 MB, `model.safetensors`) |
| Checkpoint LayoutXLM `best_model` | ✅ Còn nguyên |
| Bug Donut `max_length` | `generation_config.json` không có `max_length` → Transformers dùng mặc định 20 token |
| Log huấn luyện | ❌ Chỉ có lượt chạy test ngắn (21 step / 5 step) |
| Oracle OCR data | ✅ 784/1091 train, 168/234 test (mẫu MC-OCR) |

## Phân bổ GPU

| Session | Nội dung | GPU ước tính | Ưu tiên |
|---|---|---|---|
| **Notebook 1** | Train LayoutXLM + Inference Donut v2 + Baseline | ~3 giờ | 🔴 BẮT BUỘC |
| **Notebook 2** | Train LayoutXLM Oracle + Inference Oracle | ~3 giờ | 🟡 NÊN LÀM |
| **Notebook 3** | Train lại Donut (nếu kết quả kém) | ~18 giờ | 🟢 TÙY CHỌN |
| **Tổng** | | **~6-24 giờ** | |

> **Quan trọng:** Sau mỗi notebook, tải output về máy local → copy vào repo → chạy
> evaluation pipeline trên local → kiểm tra kết quả trước khi quyết định bước tiếp.

---

## NOTEBOOK 1: Train LayoutXLM + Inference Donut v2 (BẮT BUỘC)

**Mục đích:**
- Train lại LayoutXLM để có **log huấn luyện đầy đủ** (epoch dừng, loss curve)
- Chạy inference Donut với **`max_length=768`** (sửa bug 20 token)
- Chạy lại Baseline inference cùng điều kiện

**Cấu hình Kaggle Notebook:**
- Accelerator: **GPU T4 x2**
- Internet: **On**
- Dataset: **receipt-vn-ie-data-new** + **receipt-vn-ie-checkpoints** (chứa donut_best.zip)

### Cell 1: Khởi tạo môi trường

```python
import os, subprocess, shutil

# 1. Setup workspace
os.chdir("/kaggle/working")
subprocess.run(["rm", "-rf", "receipt-vn-ie"], check=False)

# 2. Clone code
!git clone -b main https://github.com/Duc-AnhTp/receipt-vn-ie.git
os.chdir("/kaggle/working/receipt-vn-ie")

# 3. Liên kết dữ liệu
!rm -rf data
!ln -s /kaggle/input/datasets/ducanhtp/receipt-vn-ie-data-new/data data

# 4. Cài đặt dependencies
!pip install -e . 2>&1 | tail -3
!pip install seqeval evaluate 2>&1 | tail -3
!pip install 'git+https://github.com/facebookresearch/detectron2.git' 2>&1 | tail -3
!pip install -r requirements/ocr.txt 2>&1 | tail -3
!pip install paddlepaddle-gpu 2>&1 | tail -3
!pip install "numpy<2.0.0" "albumentations<1.4.0" 2>&1 | tail -3

# 5. Tạo thư mục
!mkdir -p checkpoints/donut/receipt_ie/finetune outputs/predictions outputs/metrics

# 6. Ghi lại thông tin môi trường (cho báo cáo)
print("=" * 60)
print("THÔNG TIN MÔI TRƯỜNG")
print("=" * 60)
!python -c "import torch; print(f'PyTorch: {torch.__version__}')"
!python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!date -u
print("=" * 60)
```

### Cell 2: Khôi phục checkpoint Donut từ dataset

```python
import subprocess

# Tìm donut checkpoint trong Kaggle input
result = subprocess.run(
    ["find", "/kaggle/input", "-name", "model.safetensors", "-path", "*donut*best_model*"],
    capture_output=True, text=True
)
lines = result.stdout.strip().split("\n")

if lines and lines[0]:
    DONUT_CHECKPOINT = lines[0].rsplit("/", 1)[0]
    print(f"✅ Donut checkpoint: {DONUT_CHECKPOINT}")
    !ls -la {DONUT_CHECKPOINT}
else:
    print("❌ Không tìm thấy Donut checkpoint!")
    print("Thử giải nén từ donut_best.zip...")
    !find /kaggle/input -name "donut_best.zip" -exec unzip -o {} -d /kaggle/working/ \;
    DONUT_CHECKPOINT = "/kaggle/working/receipt-vn-ie/checkpoints/donut/receipt_ie/finetune/best_model"
    !ls -la {DONUT_CHECKPOINT}
```

> **⚠️ LƯU Ý QUAN TRỌNG TRÁNH LỖI OOM (OUT OF MEMORY) TRÊN T4 x2:**
> - **Batch Size:** Mặc định trong cấu hình sử dụng `train_batch_size: 8` per device, rất dễ gây ra lỗi tràn bộ nhớ (OOM / SIGKILL 9) trên GPU Kaggle T4. Khuyến nghị chạy với `--train_batch_size 2` và tăng `--gradient_accumulation_steps 16` tương ứng.
> - **Gradient Checkpointing:** Tuyệt đối không dùng `--gradient_checkpointing` vì lớp mô hình `LayoutLMv2ForTokenClassification` không hỗ trợ tính năng này trong thư viện Transformers.

### Cell 3: Train LayoutXLM (có log đầy đủ)

```python
import time

print("=" * 60)
print("BẮT ĐẦU HUẤN LUYỆN LAYOUTXLM")
print(f"Thời gian bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 60)

!accelerate launch --multi_gpu --num_processes 2 \
  -m receipt_ie.training.train_layoutxlm \
  --mode ocr_cache \
  --train_batch_size 2 \
  --eval_batch_size 4 \
  --gradient_accumulation_steps 16

print("=" * 60)
print(f"Thời gian kết thúc: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 60)
```

### Cell 4: Lưu log huấn luyện LayoutXLM

```python
import json, glob

# Tìm trainer_state.json mới nhất
state_files = sorted(
    glob.glob("checkpoints/layoutxlm/receipt_ie/ocr_cache/*/trainer_state.json"),
    key=os.path.getmtime, reverse=True
)

if state_files:
    with open(state_files[0]) as f:
        state = json.load(f)
    
    print(f"✅ Trainer state: {state_files[0]}")
    print(f"   Global step: {state.get('global_step')}")
    print(f"   Best metric: {state.get('best_metric')}")
    print(f"   Best checkpoint: {state.get('best_model_checkpoint')}")
    print(f"   Epoch: {state.get('epoch')}")
    print(f"   Log entries: {len(state.get('log_history', []))}")
    
    # Copy trainer_state ra output để dễ tải về
    !mkdir -p outputs/training_logs
    !cp {state_files[0]} outputs/training_logs/layoutxlm_trainer_state.json
    
    # In loss history tóm tắt
    log_history = state.get("log_history", [])
    eval_logs = [l for l in log_history if "eval_loss" in l]
    print(f"\n   Eval checkpoints: {len(eval_logs)}")
    if eval_logs:
        print(f"   First eval loss: {eval_logs[0].get('eval_loss', 'N/A')}")
        print(f"   Last eval loss: {eval_logs[-1].get('eval_loss', 'N/A')}")
        best_eval = min(eval_logs, key=lambda x: x.get("eval_loss", float("inf")))
        print(f"   Best eval loss: {best_eval.get('eval_loss')} at step {best_eval.get('step')}")
else:
    print("❌ Không tìm thấy trainer_state.json!")
```

### Cell 5: Inference LayoutXLM mới

```python
import time

LAYOUTXLM_CKPT = "checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model"

print(f"Inference LayoutXLM từ: {LAYOUTXLM_CKPT}")
print(f"Bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

!python -m receipt_ie.inference.infer_layoutxlm \
  --checkpoint {LAYOUTXLM_CKPT} \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/layoutxlm_test.jsonl \
  --allow_overwrite

print(f"Kết thúc: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
```

### Cell 6: Inference Donut v2 (SỬA BUG max_length)

```python
import time

# Sử dụng checkpoint Donut đã tìm ở Cell 2
print(f"Inference Donut từ: {DONUT_CHECKPOINT}")
print(f"generation_max_length: 768 (SỬA BUG từ 20 token)")
print(f"Bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

!python -m receipt_ie.inference.infer_donut \
  --checkpoint {DONUT_CHECKPOINT} \
  --generation_max_length 768 \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/donut_test.jsonl \
  --allow_overwrite

print(f"Kết thúc: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
```

### Cell 7: Inference Baseline

```python
!python -m receipt_ie.inference.infer_baseline \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/baseline_test.jsonl \
  --allow_overwrite
```

### Cell 8: Đóng gói output

```python
import os
os.chdir("/kaggle/working")

# Đóng gói toàn bộ predictions + logs + checkpoint mới
!zip -r retrain_v2_output.zip \
  receipt-vn-ie/outputs/predictions/ \
  receipt-vn-ie/outputs/training_logs/ \
  receipt-vn-ie/checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model/

# Đóng gói riêng checkpoint LayoutXLM mới (để dùng cho Notebook 2)  
!zip -r layoutxlm_best_v2.zip \
  receipt-vn-ie/checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model/

print("=" * 60)
print("✅ HOÀN TẤT NOTEBOOK 1")
print("Tải về: retrain_v2_output.zip và layoutxlm_best_v2.zip")
print("=" * 60)
```

---
## NOTEBOOK 2: Oracle OCR — Trả lời RQ3 (NÊN LÀM)

**Mục đích:** Train LayoutXLM với ground-truth OCR → đo "OCR gap" → trả lời RQ3

**Cấu hình Kaggle:** Tương tự Notebook 1

### Cell 1: Khởi tạo (giống Notebook 1)

```python
import os

os.chdir("/kaggle/working")
!rm -rf receipt-vn-ie

!git clone -b main https://github.com/Duc-AnhTp/receipt-vn-ie.git
os.chdir("/kaggle/working/receipt-vn-ie")

# Liên kết dữ liệu
!rm -rf data
!ln -s /kaggle/input/datasets/ducanhtp/receipt-vn-ie-data-new/data data

!pip install -e . 2>&1 | tail -3
!pip install seqeval evaluate 2>&1 | tail -3
!pip install 'git+https://github.com/facebookresearch/detectron2.git' 2>&1 | tail -3
!pip install -r requirements/ocr.txt 2>&1 | tail -3
!pip install paddlepaddle-gpu 2>&1 | tail -3
!pip install "numpy<2.0.0" "albumentations<1.4.0" 2>&1 | tail -3

!mkdir -p checkpoints outputs/predictions outputs/training_logs
```

> **⚠️ LƯU Ý QUAN TRỌNG TRÁNH LỖI OOM (OUT OF MEMORY) TRÊN T4 x2:**
> - **Batch Size:** Mặc định trong cấu hình có thể sử dụng `train_batch_size: 8` per device, dễ gây ra lỗi tràn bộ nhớ (OOM / SIGKILL 9) trên GPU Kaggle T4. Khuyến nghị chạy với `--train_batch_size 2` (hoặc `--train_batch_size 4` nếu dữ liệu nhẹ) và tăng `--gradient_accumulation_steps` tương ứng để đảm bảo ổn định.
> - **Gradient Checkpointing:** Tuyệt đối không dùng `--gradient_checkpointing` vì lớp mô hình `LayoutLMv2ForTokenClassification` không hỗ trợ tính năng này trong thư viện Transformers và sẽ gây crash ngay lập tức.

### Cell 2: Train LayoutXLM Oracle OCR

```python
import time

print("=" * 60)
print("HUẤN LUYỆN LAYOUTXLM ORACLE OCR")
print(f"Bắt đầu: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("Dữ liệu: chỉ dùng mẫu MC-OCR có oracle_ocr (784 train)")
print("=" * 60)

!accelerate launch --multi_gpu --num_processes 2 \
  -m receipt_ie.training.train_layoutxlm \
  --mode oracle_ocr \
  --train_batch_size 2 \
  --eval_batch_size 4 \
  --gradient_accumulation_steps 16

print(f"Kết thúc: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
```

### Cell 3: Lưu log huấn luyện Oracle

```python
import json, glob

state_files = sorted(
    glob.glob("checkpoints/layoutxlm/receipt_ie/oracle_ocr/*/trainer_state.json"),
    key=os.path.getmtime, reverse=True
)

if state_files:
    with open(state_files[0]) as f:
        state = json.load(f)
    print(f"✅ Oracle trainer state: global_step={state.get('global_step')}, best_metric={state.get('best_metric')}")
    !mkdir -p outputs/training_logs
    !cp {state_files[0]} outputs/training_logs/layoutxlm_oracle_trainer_state.json
```

### Cell 4: Inference Oracle LayoutXLM

> **⚠️ LƯU Ý:** Inference oracle cần đọc trường `oracle_ocr` từ test.jsonl thay vì `ocr_cache`.
> Pipeline hiện tại đã tích hợp sẵn script này.

```python
# Script inference oracle

!python -m receipt_ie.inference.infer_layoutxlm_oracle \
  --checkpoint checkpoints/layoutxlm/receipt_ie/oracle_ocr/best_model \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/layoutxlm_oracle_test.jsonl
```

### Cell 5: Đóng gói

```python
import os
os.chdir("/kaggle/working")

!zip -r oracle_output.zip \
  receipt-vn-ie/outputs/predictions/layoutxlm_oracle_test.jsonl \
  receipt-vn-ie/outputs/training_logs/ \
  receipt-vn-ie/checkpoints/layoutxlm/receipt_ie/oracle_ocr/best_model/

print("✅ HOÀN TẤT NOTEBOOK 2 — Tải về oracle_output.zip")
```

---

## NOTEBOOK 3: Train lại Donut (TÙY CHỌN)

> **Chỉ chạy nếu:** Inference Donut v2 (Notebook 1) cho kết quả rất kém
> (ví dụ: Macro EM < 15%). Nếu EM > 20% thì checkpoint cũ đã đủ tốt,
> không cần train lại.

**Thời gian ước tính:** 15-20 giờ (150 epoch, early stop có thể dừng sớm)

### Cell 1: Khởi tạo + Resume training

```python
import os

os.chdir("/kaggle/working")
!rm -rf receipt-vn-ie

!git clone -b main https://github.com/Duc-AnhTp/receipt-vn-ie.git
os.chdir("/kaggle/working/receipt-vn-ie")

# Liên kết dữ liệu
!rm -rf data
!ln -s /kaggle/input/datasets/ducanhtp/receipt-vn-ie-data-new/data data

!pip install -e . 2>&1 | tail -3
!pip install seqeval evaluate 2>&1 | tail -3
!pip install -r requirements/ocr.txt 2>&1 | tail -3
!pip install paddlepaddle-gpu 2>&1 | tail -3
!pip install "numpy<2.0.0" "albumentations<1.4.0" 2>&1 | tail -3

!mkdir -p checkpoints outputs
```

### Cell 2: Train Donut

```python
import time

print(f"Bắt đầu train Donut: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

!accelerate launch --multi_gpu --num_processes 2 \
  --mixed_precision fp16 \
  -m receipt_ie.training.train_donut \
  --mode finetune

print(f"Kết thúc: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
```

### Cell 3: Inference + Đóng gói

```python
!python -m receipt_ie.inference.infer_donut \
  --checkpoint checkpoints/donut/receipt_ie/finetune/best_model \
  --generation_max_length 768 \
  --test_jsonl data/processed/test.jsonl \
  --output_jsonl outputs/predictions/donut_test_retrained.jsonl

import os
os.chdir("/kaggle/working")
!zip -r donut_retrained_output.zip \
  receipt-vn-ie/outputs/predictions/donut_test_retrained.jsonl \
  receipt-vn-ie/checkpoints/donut/receipt_ie/finetune/best_model/

print("✅ HOÀN TẤT NOTEBOOK 3")
```

---

## Sau khi tải output về local

### Bước 1: Copy predictions vào repo

```powershell
# Giải nén output
Expand-Archive retrain_v2_output.zip -DestinationPath .

# Copy predictions vào đúng vị trí
Copy-Item "receipt-vn-ie\outputs\predictions\*.jsonl" `
  "D:\Users\Documents\HUCE\Thi_Giac_May_Tinh\DoAnTGMT\receipt-vn-ie\outputs\predictions\" -Force

# Copy training logs
Copy-Item "receipt-vn-ie\outputs\training_logs\*" `
  "D:\Users\Documents\HUCE\Thi_Giac_May_Tinh\DoAnTGMT\receipt-vn-ie\outputs\training_logs\" -Force
```

### Bước 2: Chạy evaluation pipeline

```bash
cd D:\Users\Documents\HUCE\Thi_Giac_May_Tinh\DoAnTGMT\receipt-vn-ie

# Regenerate metrics, plots, LaTeX tables
bash scripts/05_evaluate_all.sh

# Hoặc trên Windows:
python -m receipt_ie.metrics.summarize_outputs ^
  --gold data/processed/test.jsonl ^
  --pred outputs/predictions/baseline_test.jsonl ^
       outputs/predictions/layoutxlm_test.jsonl ^
       outputs/predictions/donut_test.jsonl ^
  --metrics_output outputs/metrics/combined_metrics.json ^
  --latency_output outputs/metrics/latency_by_method.json

python scripts/build_artifact_manifest.py
python -m receipt_ie.metrics.comparison_analysis
python scripts/plot_results.py
python scripts/generate_report_artifacts.py
```

### Bước 3: Kiểm tra kết quả

```bash
# Xem metrics nhanh
python -c "
import json
m = json.load(open('outputs/metrics/combined_metrics.json'))
for method in ['baseline', 'layoutxlm', 'donut']:
    if method in m:
        po = m[method].get('present_only', {}).get('macro', {})
        print(f'{method}: EM={po.get(\"EM\",0)*100:.1f}% NES={po.get(\"NES\",0)*100:.1f}%')
"
```

### Bước 4: Cập nhật báo cáo

Sau khi `scripts/generate_report_artifacts.py` chạy xong, các file trong
`report/generated/` sẽ tự cập nhật. Bạn cần sửa tay:

1. **`artifact_validity.yaml`** — cập nhật Donut thành `valid_for_main_comparison: true`
2. **`00_abstract.tex`** — thêm kết quả Donut mới
3. **`05_methodology.tex`** — xóa/sửa warning box Donut
4. **`06_experiments.tex`** — thêm thông tin training log
5. **`07_results.tex`** — thêm Donut vào bảng chính, thêm bảng Oracle (nếu có)
6. **`08_conclusion.tex`** — cập nhật trả lời RQ1 (có Donut), RQ3 (có Oracle)

### Bước 5: Commit và push

```bash
git add -A
git commit -m "retrain v2: fix donut inference, add training logs, oracle ocr"
git push origin main
```



---

## Checklist tổng hợp

### Trước khi chạy Kaggle:
- [ ] Upload `donut_best.zip` lên Kaggle Dataset (nếu chưa có)

### Notebook 1 (BẮT BUỘC):
- [ ] Train LayoutXLM → lưu `trainer_state.json`
- [ ] Inference LayoutXLM mới
- [ ] Inference Donut v2 (`max_length=768`)
- [ ] Inference Baseline
- [ ] Tải `retrain_v2_output.zip` về

### Notebook 2 (NÊN LÀM):
- [ ] Train LayoutXLM Oracle OCR
- [ ] Inference Oracle
- [ ] Tải `oracle_output.zip` về

### Sau khi tải về local:
- [ ] Copy predictions vào repo
- [ ] Chạy `scripts/05_evaluate_all.sh`
- [ ] Kiểm tra kết quả metrics
- [ ] Cập nhật `artifact_validity.yaml`
- [ ] Cập nhật báo cáo LaTeX
- [ ] Commit + push

### Cập nhật báo cáo:
- [ ] Cập nhật tóm tắt (`00_abstract.tex`)
- [ ] Sửa warning box Donut (`05_methodology.tex`)
- [ ] Thêm training log vào thực nghiệm (`06_experiments.tex`)
- [ ] Cập nhật bảng kết quả (`07_results.tex`) — tự động từ generated/
- [ ] Thêm bảng Oracle OCR vào kết quả (nếu có)
- [ ] Cập nhật kết luận RQ1, RQ3 (`08_conclusion.tex`)
- [ ] Cập nhật threats to validity — loại bỏ các mục đã sửa
