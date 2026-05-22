# Dataset Card — Receipt VN IE

## Tổng quan

Tập dữ liệu biên lai tiếng Việt phục vụ bài toán trích xuất 4 trường thông tin: `store_name`, `date`, `total`, `address`.

## Nguồn dữ liệu

### MC-OCR 2021

- **Mô tả**: 2.436 biên lai tiếng Việt chụp bằng thiết bị di động từ cuộc thi MC-OCR 2021.
- **Nhãn gốc**: `SELLER`, `SELLER_ADDRESS`, `TIMESTAMP`, `TOTAL_COST` (mapping sang `store_name`, `address`, `date`, `total`).
- **Annotation level**: `json_and_boxes` (có polygon + text).
- **Lưu ý**: Tập public test của MC-OCR **không có ground-truth labels** → chỉ sử dụng phần train có nhãn, tự chia nội bộ.
- **Đường dẫn**: `data/raw/mc_ocr_2021/`

### Tự thu thập (Self-collected)

- **Mô tả**: ~500 ảnh biên lai tiếng Việt tự chụp từ nhiều cửa hàng, siêu thị.
- **Annotation tool**: Label Studio.
- **Annotation level**: `json_and_boxes` (tối thiểu 200–300 ảnh) hoặc `json_only`.
- **Đường dẫn**: `data/raw/self_collected/`

### CORD v2

- **Mô tả**: Consolidated Receipt Dataset v2 từ Naver CLOVA.
- **Vai trò**: Chỉ dùng làm warm-up/pretrain cho Donut. **Không đưa vào test chính**.
- **Task token riêng**: `<s_cord_receipt_parse>` (tách biệt với task chính `<s_receipt_ie>`).
- **Đường dẫn**: `data/raw/cord_v2/`

## Schema thống nhất (Unified JSONL)

Mỗi dòng trong file JSONL chứa:

| Trường | Kiểu | Mô tả |
|---|---|---|
| `id` | string | ID duy nhất của mẫu |
| `group_id` | string | ID nhóm (cùng bill chụp nhiều góc) |
| `store_group` | string | Nhóm cửa hàng (phục vụ stress-test split) |
| `source` | string | `mc_ocr_2021` / `self_collected` / `cord_v2` |
| `image_path` | string | Đường dẫn tương đối tới ảnh |
| `width`, `height` | int | Kích thước ảnh gốc (pixels) |
| `annotation_level` | string | `json_only` hoặc `json_and_boxes` |
| `raw_target` | object | Nhãn gốc chưa chuẩn hóa (4 trường) |
| `target` | object | Nhãn đã chuẩn hóa (4 trường) |
| `field_boxes` | object | Bounding boxes [x0, y0, x1, y1] cho mỗi trường |
| `field_polygons` | object | Polygons cho mỗi trường |
| `ocr_cache_path` | string | Đường dẫn tới file OCR cache |

## Chiến lược chia dữ liệu

### Main Split (Chính)
- **Tỉ lệ**: 70% Train / 15% Val / 15% Test
- **Group split**: Theo `group_id` để tránh leak cùng bill chụp nhiều góc
- **Stratify**: Theo `source`
- **Anti-leak**: MD5 hash kiểm tra trùng ảnh giữa các tập

### Stress-Test Split (Phụ)
- **Phương pháp**: `store_group` held-out hoàn toàn
- **Mục đích**: Đánh giá generalize sang cửa hàng/layout mới

## Quy tắc chuẩn hóa

Xem chi tiết tại [normalization_rules.md](normalization_rules.md).

## Thống kê

*Sẽ được cập nhật sau khi convert và split dữ liệu.*

| Tập | MC-OCR | Self-collected | Tổng |
|---|---|---|---|
| Train | — | — | — |
| Val | — | — | — |
| Test | — | — | — |
