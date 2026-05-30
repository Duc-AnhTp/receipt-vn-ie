#!/usr/bin/env bash
# Bootstrap script chạy trên Kaggle để cài đặt môi trường và copy data
set -euo pipefail

echo "=== Bắt đầu cài đặt môi trường Receipt VN IE trên Kaggle ==="

# Cài đặt các requirements
pip install -e .
pip install -r requirements/app.txt
pip install -r requirements/ocr.txt
pip install paddlepaddle-gpu

# Setup data từ Kaggle Dataset
# Giả định dataset của Kaggle đặt tại /kaggle/input/receipt-vn-ie-data
# Dữ liệu chứa file zip hoặc folder data
KAGGLE_INPUT_DIR="/kaggle/input/receipt-vn-ie-data"

if [ -d "$KAGGLE_INPUT_DIR" ]; then
  echo "Tìm thấy thư mục dữ liệu Kaggle: $KAGGLE_INPUT_DIR"
  mkdir -p data
  
  # Nếu có file zip
  if [ -f "$KAGGLE_INPUT_DIR/receipt_dataset.zip" ]; then
    echo "Đang giải nén dữ liệu..."
    unzip -q "$KAGGLE_INPUT_DIR/receipt_dataset.zip" -d data/
  else
    echo "Copying data folder..."
    cp -r "$KAGGLE_INPUT_DIR"/* data/
  fi
  echo "Dữ liệu đã được chuẩn bị thành công trong thư mục data/"
else
  echo "[CẢNH BÁO] Không tìm thấy thư mục /kaggle/input/receipt-vn-ie-data. Vui lòng kiểm tra lại cấu trúc Kaggle Input."
fi
