# Hướng Dẫn Chi Tiết Huấn Luyện Trên Kaggle GPU T4x2

Dự án hiện tại đã **hoàn toàn sẵn sàng và đủ điều kiện** để đưa lên huấn luyện trên môi trường **Kaggle GPU T4x2** (2 GPU Nvidia Tesla T4 song song). Lỗi constructor dataset đã được sửa, data augmentation cho cả hai mô hình đã được kích hoạt thực sự, và toàn bộ ảnh đã được tiền xử lý thành công.

Dưới đây là quy trình từng bước chi tiết để bạn đóng gói, tải lên và khởi chạy huấn luyện trên Kaggle.

---

## 1. Chuẩn Bị Dữ Liệu và Mã Nguồn Tại Local

Có 2 cách để bạn đưa mã nguồn lên Kaggle: **Sử dụng GitHub (khuyên dùng)** hoặc **Nén file Zip thủ công**.

### CÁCH 1: Sử Dụng GitHub (Khuyên dùng - Cập nhật siêu nhanh)
Vì file cấu hình `.gitignore` của dự án đã chặn toàn bộ các thư mục dữ liệu nặng và mô hình (`data/`, `checkpoints/`, `outputs/`, `*.zip`), thư mục mã nguồn đẩy lên GitHub sẽ cực kỳ nhẹ (chỉ khoảng vài trăm KB), giúp bạn đồng bộ chỉ trong 1 giây.

#### Bước 1.1: Push code từ máy local lên GitHub
Nếu bạn chưa đưa dự án lên GitHub, hãy chạy các lệnh sau tại terminal của máy local:
```bash
# Khởi tạo git và commit code
git init
git add .
git commit -m "Configure Receipt VN IE project for Kaggle"

# Tạo repo mới trên GitHub (Ví dụ đặt tên là receipt-vn-ie)
# Thêm remote và đẩy code lên nhánh main
git remote add origin https://github.com/Duc-AnhTp/receipt-vn-ie.git
git branch -M main
git push -u origin main
```

#### Bước 1.2: Chuẩn bị ảnh và dữ liệu (Vẫn nén ZIP tải lên Kaggle Dataset)
Vì dữ liệu ảnh rất nặng và không đưa lên GitHub, bạn hãy nén thư mục dữ liệu lại để upload làm Kaggle Dataset:
*   *Lệnh nén nhanh trên Windows (PowerShell) bỏ qua thư mục ảnh raw:*
    ```powershell
    Compress-Archive -Path data/processed, data/interim -DestinationPath data.zip
    ```

---

### CÁCH 2: Nén File Zip Mã Nguồn Thủ Công
Nếu không muốn dùng GitHub, bạn hãy nén code lại như sau:
*   **Các thư mục cần nén:** `src/`, `configs/`, `scripts/`, `pyproject.toml`.
*   *Lệnh nén nhanh trên Windows (PowerShell):*
    ```powershell
    Compress-Archive -Path src, configs, scripts, pyproject.toml -DestinationPath code.zip
    ```
*   **Đóng gói dữ liệu:** Làm tương tự như Bước 1.2 để có file `data.zip`.

---

## 2. Thiết Lập Trên Kaggle

### Bước 2.1: Tạo Kaggle Dataset chứa dữ liệu
1. Truy cập vào Kaggle -> chọn **Datasets** -> chọn **New Dataset**.
2. Đặt tên dataset là `receipt-vn-ie-data`.
3. Tải file `data.zip` (và file `code.zip` nếu dùng Cách 2) lên hệ thống và nhấn **Create**.

### Bước 2.2: Khởi tạo Kaggle Notebook
1. Chọn **Code** -> chọn **New Notebook**.
2. Ở bảng cấu hình bên phải (Notebook options):
   * **Accelerator:** Chọn `GPU T4 x2` (Kích hoạt 2 GPU song song).
   * **Internet:** Bật trạng thái **On** (để tải thư viện và pretrained model).
3. Nhấp vào nút **Add Data** -> Tìm kiếm dataset `receipt-vn-ie-data` bạn vừa tạo và add vào notebook.

---

## 3. Khởi Chạy Huấn Luyện Trong Kaggle Notebook

Hãy tạo các cell mới trong Notebook và chạy các lệnh tương ứng dưới đây:

### Bước 3.1: Nạp mã nguồn và dữ liệu vào Kaggle
Tùy thuộc vào việc bạn chọn nạp code qua **GitHub** (Cách 1) hay **File Zip** (Cách 2):

#### A. Nếu dùng Cách 1 (GitHub)
*   **Nếu Repo của bạn là Public:**
    ```bash
    git clone https://github.com/Duc-AnhTp/receipt-vn-ie.git /kaggle/working/receipt-vn-ie
    cd /kaggle/working/receipt-vn-ie
    ```
*   **Nếu Repo của bạn là Private (Bảo mật):**
    Hãy tạo một **Personal Access Token (PAT)** trên GitHub (vào *Settings -> Developer settings -> Personal access tokens -> Tokens (classic)* -> tích chọn quyền `repo` -> nhấn *Generate token*). Sau đó chạy clone bằng link chứa token trên Kaggle Notebook:
    ```bash
    git clone https://<MÃ_TOKEN_CỦA_BẠN>@github.com/Duc-AnhTp/receipt-vn-ie.git /kaggle/working/receipt-vn-ie
    cd /kaggle/working/receipt-vn-ie
    ```

*   **Cách PULL cập nhật code cực nhanh khi sửa code ở local:**
    Mỗi khi bạn sửa code ở local, bạn chỉ cần push lên GitHub (`git push`). Sau đó, trên Kaggle Notebook bạn chỉ cần chạy cell này để cập nhật code mới trong 1 giây mà không cần upload lại file ZIP:
    ```bash
    cd /kaggle/working/receipt-vn-ie
    git pull
    ```

#### B. Nếu dùng Cách 2 (File Zip)
```bash
# Tạo thư mục làm việc chính
mkdir -p /kaggle/working/receipt-vn-ie
cd /kaggle/working/receipt-vn-ie

# Giải nén mã nguồn từ dataset
unzip -q /kaggle/input/receipt-vn-ie-data/code.zip -d .
```

#### Giải nén dữ liệu (Dành cho cả 2 cách)
```bash
# Giải nén dữ liệu từ dataset
mkdir -p data
unzip -q /kaggle/input/receipt-vn-ie-data/data.zip -d data/
```

### Bước 3.2: Cài đặt các thư viện cần thiết
```bash
# Cài đặt gói receipt_ie ở chế độ editable và các thư viện bổ sung
pip install -e .
pip install albumentations --upgrade
```

### Bước 3.3: Huấn luyện LayoutXLM trên T4x2 (Sử dụng Accelerate Distributed Training)
Để chạy mô hình song song trên cả 2 GPU T4, chúng ta sử dụng thư viện `accelerate` của Hugging Face đã được tích hợp sẵn:
```bash
# Chạy huấn luyện LayoutXLM DDP (Distributed Data Parallel) trên 2 GPU
accelerate launch --multi_gpu --num_processes 2 \
  -m receipt_ie.training.train_layoutxlm \
  --mode ocr_cache
```
*   *Lưu ý:* Quá trình huấn luyện sẽ tự động áp dụng Data Augmentation (mờ ảnh, bóng đổ, nhiễu ánh sáng) giúp nâng cao độ chính xác và lưu checkpoint tốt nhất vào `checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model`.

### Bước 3.4: Huấn luyện Donut trên T4x2 (DDP)
Sau khi LayoutXLM hoàn tất, bạn có thể chạy tiếp huấn luyện Donut:
```bash
# Chạy huấn luyện Donut trên 2 GPU
accelerate launch --multi_gpu --num_processes 2 \
  -m receipt_ie.training.train_donut \
  --mode finetune
```
*   *Lưu ý:* Donut được huấn luyện với các phép tăng cường hình học nâng cao (xoay ảnh, xoay phối cảnh) trong 30 epochs và lưu checkpoint tốt nhất vào `checkpoints/donut/receipt_ie/finetune/best_model`.

---

## 4. Tải Checkpoint Kết Quả Về Local

Sau khi quá trình huấn luyện hoàn tất, bạn cần nén các checkpoint tốt nhất lại để tải về máy local:
```bash
# Nén checkpoint LayoutXLM
zip -r layoutxlm_best.zip checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model

# Nén checkpoint Donut
zip -r donut_best.zip checkpoints/donut/receipt_ie/finetune/best_model
```
Nhấp vào tab **Files** ở bảng bên trái của Kaggle Notebook, tìm file `layoutxlm_best.zip` và `donut_best.zip` nằm dưới thư mục `/kaggle/working/`, chọn **Download** để tải về máy của bạn!
