# Hướng dẫn Mở rộng Tập dữ liệu Huấn luyện bằng Label Studio

Tài liệu này hướng dẫn nhóm quy trình thu thập thêm ảnh hóa đơn thực tế từ các cửa hàng khác nhau ở Việt Nam, cách sử dụng công cụ **Label Studio** để gán nhãn, và cách chuyển đổi tích hợp dữ liệu mới này vào luồng huấn luyện của dự án.

---

## Bước 1: Thu thập ảnh hóa đơn thực tế
- **Số lượng khuyến nghị:** Bổ sung thêm khoảng 100 - 200 mẫu hóa đơn từ các quán cafe, nhà hàng, hiệu thuốc, siêu thị tiện lợi khác nhau ở Việt Nam.
- **Tiêu chuẩn chụp ảnh:**
  - Chụp trong nhiều điều kiện ánh sáng (trong nhà, ngoài trời, bóng đổ nhẹ).
  - Nghiêng góc camera nhẹ (độ nghiêng từ -10 đến 10 độ) để kiểm tra thuật toán tự động căn thẳng (`rectify_document`).
  - Đa dạng các layout hóa đơn: in nhiệt, in kim, hóa đơn khổ dọc dài, hóa đơn có nhiều bảng kê số tiền...

---

## Bước 2: Thiết lập và Gán nhãn bằng Label Studio

### 2.1. Cài đặt & Khởi động Label Studio
Cài đặt Label Studio bằng pip trong môi trường Python của bạn:
```bash
pip install label-studio
label-studio
```
Trình duyệt sẽ tự động mở trang quản trị tại địa chỉ `http://localhost:8080`.

### 2.2. Tạo Project & Cấu hình Gán nhãn
1. Bấm **Create Project**, đặt tên là `Receipt-VN-IE-SelfCollected`.
2. Trong tab **Labeling Setup**, chọn **Custom Template** và dán đoạn mã cấu hình XML dưới đây để thiết lập đồng thời gán nhãn hộp bao (`rectanglelabels`) và văn bản trích xuất (`textarea`):

```xml
<View>
  <Image name="image" value="$image"/>
  
  <!-- Nhãn hộp bao tương ứng với 4 thực thể -->
  <RectangleLabels name="label" toName="image">
    <Label value="store_name" background="#EF4444"/>
    <Label value="date" background="#22C55E"/>
    <Label value="total" background="#F97316"/>
    <Label value="address" background="#A855F7"/>
  </RectangleLabels>
  
  <!-- Vùng nhập văn bản tương ứng cho từng hộp bao -->
  <TextArea name="transcription" toName="image" editable="true" perRegion="true" required="true"/>
</View>
```

3. Bấm **Save** để hoàn tất cấu hình.

### 2.3. Tải ảnh lên và Tiến hành gán nhãn
1. Import các ảnh hóa đơn mới thu thập vào project.
2. Với mỗi ảnh, kéo thả các hộp bao tương ứng vào vùng văn bản của 4 trường thông tin.
3. Sau khi vẽ xong một hộp bao, bấm vào hộp bao đó để nhập văn bản thực tế trong ô **transcription** (bên dưới hoặc bên cạnh ảnh).
4. Nhấn **Submit** để lưu nhãn cho từng ảnh.

---

## Bước 3: Xuất dữ liệu và Tích hợp vào Dự án

### 3.1. Xuất dữ liệu từ Label Studio
Sau khi gán nhãn xong toàn bộ ảnh:
1. Vào tab **Export** trong dự án Label Studio.
2. Chọn định dạng **JSON** và tải tệp về.
3. Đặt tệp JSON này vào thư mục dự án tại đường dẫn: `data/raw/self_collected/label_studio.json`.
4. Copy toàn bộ ảnh đã gán nhãn vào thư mục: `data/raw/self_collected/images/`.

### 3.2. Chạy kịch bản chuyển đổi dữ liệu
Chạy script `convert_labelstudio.py` có sẵn để tự động chuyển đổi file JSON của Label Studio sang định dạng JSONL hợp nhất:
```bash
python -m receipt_ie.data.convert_labelstudio --json_path data/raw/self_collected/label_studio.json --images_dir data/raw/self_collected/images --output_jsonl data/interim/self_unified.jsonl
```

Kịch bản này sẽ:
- Trích xuất tọa độ bounding box, chuẩn hóa về dải $[0, 1000]$.
- Chuẩn hóa chuẩn mực văn bản (loại bỏ khoảng trắng thừa, chuẩn hóa định dạng ngày tháng và tiền tệ).
- Lưu kết quả vào tệp `data/interim/self_unified.jsonl`.

---

## Bước 4: Chia tập dữ liệu & Huấn luyện lại mô hình

1. **Trộn dữ liệu mới với dữ liệu MC-OCR và chia tập:**
   Chạy script phân tách dữ liệu để cập nhật lại các tệp train/val/test:
   ```bash
   python -m receipt_ie.data.split_data
   ```
2. **Huấn luyện lại mô hình:**
   Đẩy các tệp dữ liệu đã cập nhật lên Kaggle và chạy lại kịch bản huấn luyện để cải thiện độ chính xác thực tế:
   - Huấn luyện Donut:
     ```bash
     python -m receipt_ie.training.train_donut --config configs/donut_mcocr.yaml
     ```
   - Huấn luyện LayoutXLM:
     ```bash
     python -m receipt_ie.training.train_layoutxlm --config configs/layoutxlm_mcocr.yaml
     ```
