# Hướng dẫn gán nhãn dữ liệu (Annotation Guideline)

Tài liệu này hướng dẫn cách thức gán nhãn (bounding box và transcription) cho tập dữ liệu biên lai tự thu thập nhằm đảm bảo tính đồng nhất cao nhất trước khi huấn luyện mô hình.

## 1. Quy tắc gán nhãn chung
- Đối với mỗi trường thông tin cần trích xuất, cần gán nhãn **bounding box** bao quanh từ hoặc dòng chữ chứa thông tin đó.
- Nhập văn bản chính xác xuất hiện trong ảnh (chữ hoa, chữ thường, dấu câu, lỗi chính tả gốc đều giữ nguyên) vào phần transcription. Việc chuẩn hóa sẽ được xử lý riêng bởi code tiền xử lý, không chuẩn hóa thủ công khi gán nhãn.
- Đối với các ảnh tự chụp, tối thiểu **200–300 ảnh** cần được gán đầy đủ nhãn bounding boxes cho cả 4 trường (`json_and_boxes`). Các ảnh còn lại có thể chỉ cần gán nhãn JSON đầu ra (`json_only`).

## 2. Quy tắc gán nhãn cụ thể cho từng trường

### 2.1. store_name (Tên cửa hàng)
- **Vùng gán nhãn (box)**: Bao quanh tên thương mại hoặc logo văn bản của cửa hàng/siêu thị (ví dụ: `Co.opmart`, `Bách Hóa Xanh`, ` Highland Coffee`).
- **Phạm vi gán**: Chỉ lấy tên của đơn vị trực tiếp bán hàng.
- **Không gán**:
  - Không gán mã số thuế (MST) hoặc địa chỉ làm tên cửa hàng.
  - Không gán các cụm từ chung chung như "HÓA ĐƠN THANH TOÁN", "HÓA ĐƠN BÁN LẺ".

### 2.2. date (Ngày tháng giao dịch)
- **Vùng gán nhãn (box)**: Chỉ bao quanh cụm ký tự ngày tháng năm (ví dụ: `09/08/2020`, `22-05-2024`, `2024.05.21`).
- **Không gán**:
  - Không bao gồm phần giờ phút giây vào bounding box của ngày nếu có thể tách rời. Ví dụ: trong cụm `09/08/2020 15:30`, chỉ vẽ box xung quanh `09/08/2020`.
  - Không gán các từ khóa dẫn dắt như "Ngày:", "Date:", "Ngày in:".

### 2.3. total (Tổng tiền thanh toán)
- **Vùng gán nhãn (box)**: Bao quanh chuỗi số hiển thị tổng số tiền thực tế khách hàng phải trả (ví dụ: `115.000`, `115,000đ`, `115 000`).
- **Độ ưu tiên trích xuất**:
  1. Tổng thanh toán sau thuế/chiết khấu (Thực tế phải trả).
  2. Tổng cộng / Total / Amount / Thành tiền.
- **Không gán**:
  - Tiền tạm tính (subtotal) khi chưa cộng VAT hoặc phí dịch vụ.
  - Tiền khách đưa (Cash), tiền thừa trả lại (Change/Refund).
  - Không gán từ khóa dẫn dắt như "Tổng cộng:", "Total:".

### 2.4. address (Địa chỉ cửa hàng)
- **Vùng gán nhãn (box)**: Bao quanh toàn bộ các dòng chữ ghi địa chỉ của chi nhánh/cửa hàng thực hiện giao dịch.
- **Xử lý địa chỉ nhiều dòng**: Vẽ các bounding box riêng cho từng dòng và ghép nối chúng theo thứ tự đọc tự nhiên từ trên xuống dưới.
- **Không gán**:
  - Số điện thoại chi nhánh, website, hotline, email, số hóa đơn nằm lân cận.
  - Từ khóa dẫn dắt như "Địa chỉ:", "Address:".

## 3. Quy trình kiểm tra chất lượng nhãn (QA/QC)
Sau khi kết thúc quá trình gán nhãn, thực hiện kiểm tra ngẫu nhiên ít nhất 50 mẫu để phát hiện:
- Các box bị lệch, cắt mất một phần ký tự.
- Bị nhầm lẫn nhãn giữa các trường (ví dụ: gán box address vào nhãn store_name).
- Sai sót gõ chữ ở phần transcription của box.
