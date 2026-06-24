"""
Audit Coordinate Scaling Issue

Mục đích:
Báo cáo đã ghi nhận 4 mẫu test có sai lệch hệ tọa độ ảnh/OCR trong bảng sensitivity check.
Script này đóng vai trò như một checklist (hoặc issue) để theo dõi và hướng dẫn
quá trình trace thủ công nguyên nhân sai lệch tọa độ.

Hướng dẫn Audit:
1. Lấy danh sách 4 ID bị lỗi (từ danh sách ngoại trừ trong scripts đánh giá).
2. Trace luồng dữ liệu của Bounding Box qua các bước sau:

Bước A: OCR Cache Generation (src/receipt_ie/ocr/build_ocr_cache.py)
- Mở file ảnh gốc, kiểm tra kích thước gốc (width, height).
- Xem xét OCR trả về tọa độ hộp bao theo hệ tọa độ nào (pixel gốc hay chuẩn hóa?).
- Kiểm tra xem lúc lưu vào cache_coordinates.py, tọa độ có bị resize hay scale không.

Bước B: Dataset Preparation (src/receipt_ie/models/layoutxlm_dataset.py)
- Trong hàm __getitem__, kiểm tra cách tọa độ được chuyển đổi sang chuẩn 0-1000 của LayoutXLM.
- `scale_x = 1000 / width`, `scale_y = 1000 / height`. Width và Height ở đây lấy từ đâu? 
  Từ metadata của ảnh gốc hay từ ảnh đã resize qua preprocessor?

Bước C: Inference (src/receipt_ie/inference/pipeline.py)
- Kiểm tra lúc predict, output token bboxes được ánh xạ ngược lại ảnh như thế nào.
- Đảm bảo tọa độ không bị nhân/chia sai tỷ lệ hai lần (double scaling).

Trạng thái: TODO
"""

def main():
    print("Vui lòng đọc docstring trong file để thực hiện audit thủ công 4 mẫu sai lệch tọa độ.")

if __name__ == "__main__":
    main()
