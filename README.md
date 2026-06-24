<h1 align="center">Receipt VN IE — Trích Xuất Thông Tin Biên Lai Tiếng Việt</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

Dự án này thực hiện việc nhận dạng và trích xuất thông tin (Information Extraction) từ ảnh biên lai, hóa đơn tiếng Việt. Hệ thống tiến hành so sánh hai kiến trúc học sâu phổ biến hiện nay:
- **Donut**: Mô hình End-to-End OCR-free.
- **VietOCR + LayoutXLM**: Hệ thống pipeline đa mô-đun kết hợp OCR và hiểu biết bố cục (layout-aware).

## ✨ Tính Năng Chính
- **Trích xuất 4 trường thông tin thiết yếu**: Tên cửa hàng (`store_name`), Ngày (`date`), Tổng tiền (`total`), và Địa chỉ (`address`).
- **So sánh mô hình chuyên sâu**: Cung cấp pipeline hoàn chỉnh để huấn luyện, đánh giá và chẩn đoán độ trễ (latency), hiện tượng model collapse.
- **Metric đánh giá tự động**: Tích hợp các thang đo Exact Match (EM), Normalized Edit Similarity (NES), và Character Error Rate (CER).
- **Giao diện Web thân thiện**: Tích hợp Gradio App để upload ảnh và chạy thử inference trực tiếp.

## 🎯 Trường Thông Tin Trích Xuất

| Trường | Định dạng | Ví dụ thực tế |
|---|---|---|
| `store_name` | Chuỗi Unicode NFC | MINIMART ANAN |
| `date` | `YYYY-MM-DD` | 2020-08-09 |
| `total` | Chuỗi số nguyên | 115000 |
| `address` | Chuỗi Unicode NFC | Chợ Sủi Phú Thị Gia Lâm |

## 📂 Cấu Trúc Dự Án

```
receipt-vn-ie/
├── configs/          # YAML configs cho dataset, OCR, model, web app
├── data/             # Thư mục lưu trữ dữ liệu (Raw → Interim → Processed)
├── docs/             # Tài liệu chuẩn hóa và dataset card
├── src/receipt_ie/   # Source code chính của hệ thống
│   ├── app/          # Giao diện Gradio Web Demo
│   ├── baseline/     # Rule-based extractor (luật)
│   ├── data/         # Xử lý, chuẩn hóa, chia split dataset
│   ├── inference/    # Pipeline inference tổng hợp
│   ├── metrics/      # Code tính điểm EM, NES, CER, Error Taxonomy
│   ├── models/       # Wrappers cho LayoutXLM và Donut
│   ├── ocr/          # Tích hợp PaddleOCR (detect) + VietOCR (recognize)
│   ├── postprocess/  # Hậu xử lý kết quả
│   ├── preprocessing/# Tiền xử lý tọa độ, ảnh
│   ├── training/     # Training loop cho mô hình
│   └── visualization/# Vẽ Bounding box đối chiếu
├── tests/            # Bộ Unit tests (Pytest)
└── scripts/          # Bash/Python scripts để chạy pipeline tự động
```

## 🚀 Cài Đặt và Khởi Chạy

### 1. Chuẩn bị môi trường
Khuyến nghị sử dụng môi trường ảo (Virtual Environment) hoặc Conda:
```bash
git clone https://github.com/Duc-AnhTp/receipt-vn-ie.git
cd receipt-vn-ie

# Tạo và kích hoạt môi trường ảo (Python >= 3.10)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt Dependencies
```bash
# Cài đặt code base
pip install -e .

# Cài đặt thư viện OCR
pip install -r requirements/ocr.txt
pip install -r requirements/paddle-cpu.txt   # Sử dụng paddle-gpu.txt nếu có GPU

# Cài đặt thư viện Web App
pip install -r requirements/app.txt
```

### 3. Dữ Liệu
Sử dụng dữ liệu **MC-OCR 2021** hoặc dữ liệu tự thu thập:
- Đặt file `mcocr_train_df.csv` và thư mục ảnh vào `data/raw/mc_ocr_2021/`
- Chạy pipeline chuẩn bị dữ liệu:
```bash
# Convert dữ liệu sang cấu trúc unified JSONL
python -m receipt_ie.data.convert_mcocr
python -m receipt_ie.data.convert_labelstudio

# Chia tập train/val/test
python -m receipt_ie.data.split_data

# Kiểm tra tính hợp lệ của dữ liệu
python -m receipt_ie.data.validate_dataset --jsonl_path data/processed/train.jsonl
```

## 🖥️ Pipeline Tự Động (Scripts)

Dự án cung cấp sẵn các script tự động hóa toàn bộ quá trình thực nghiệm:
```bash
bash scripts/01_validate_data.sh      # Kiểm tra và xác thực dữ liệu
bash scripts/02_build_ocr_cache.sh    # Chạy OCR trước và lưu cache
bash scripts/03_train_layoutxlm.sh    # Huấn luyện mô hình LayoutXLM
bash scripts/04_train_donut.sh        # Huấn luyện mô hình Donut
bash scripts/05_evaluate_all.sh       # Đánh giá toàn bộ trên tập Test
bash scripts/06_error_analysis.sh     # Trích xuất và xuất file phân tích lỗi (CSV)
```

## 📊 Kết Quả Đánh Giá (Evaluation)

Đánh giá trên tập test **234 mẫu**, phân chia theo tỷ lệ 70/15/15. Seed = 42.

### 1. Kết quả Present-Only (Chỉ tính trên các trường tồn tại thực tế)
*Đây là thang đo phản ánh chính xác nhất năng lực trích xuất khi trường thông tin có mặt trên hóa đơn.*

| Phương pháp | Store EM | Date EM | Total EM | Address EM | Macro EM | Macro NES | Macro CER |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Rule-based)** | 26.34 | 75.46 | 58.06 | 1.79 | **40.41** | 55.47 | 56.81 |
| **Donut** | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.08 | 334.39 |
| **VietOCR + LayoutXLM** | 28.12 | 60.19 | 71.89 | 11.66 | **42.96** | 63.89 | 38.79 |

### 2. Kết quả All-Samples (Tính trên toàn bộ mẫu, bao gồm cả trường rỗng)
| Phương pháp | Store EM | Date EM | Total EM | Address EM | Macro EM | Macro NES | Macro CER |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (Rule-based)** | 25.21 | 74.79 | 54.27 | 4.70 | **39.74** | 54.00 | 57.71 |
| **Donut** | 4.27 | 7.69 | 7.26 | 4.70 | **5.98** | 6.05 | 318.39 |
| **VietOCR + LayoutXLM** | 29.06 | 61.11 | 71.37 | 13.68 | **43.80** | 63.74 | 38.82 |

*(Đơn vị: %. Chênh lệch của Donut do mô hình gặp hiện tượng model collapse).*

## 🌐 Chạy Web Demo (Gradio)

Dự án đi kèm một giao diện Gradio dễ sử dụng để chạy dự đoán trực tiếp:
```bash
python src/receipt_ie/app/gradio_app.py
```
Sau khi khởi chạy thành công, truy cập trình duyệt tại địa chỉ: `http://localhost:7860`.

