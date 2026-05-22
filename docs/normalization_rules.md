# Quy tắc chuẩn hóa văn bản (Normalization Rules)

Tài liệu này quy định các quy tắc lập trình dùng để xử lý và chuẩn hóa chuỗi văn bản trích xuất từ ảnh biên lai. Các hàm chuẩn hóa trong mã nguồn bắt buộc phải tuân thủ nghiêm ngặt các quy tắc này.

## 1. Chuẩn hóa tiếng Việt chung
- **Unicode NFC**: Tất cả các trường thông tin dạng văn bản (`store_name`, `address`) bắt buộc phải được chuyển về chuẩn Unicode dựng sẵn (NFC). Tránh hiện tượng lỗi font hoặc so khớp sai do Unicode tổ hợp (NFD).
- **Khoảng trắng**: Loại bỏ khoảng trắng thừa ở hai đầu chuỗi. Chuyển các nhóm khoảng trắng liên tục (bao gồm dấu cách cứng `\u00a0`, tab `\t`, xuống dòng `\n`) thành một khoảng trắng duy nhất `" "`.
- **Ký tự đặc biệt**: Loại bỏ các dấu câu, ký hiệu thừa ở đầu và cuối chuỗi như `: ; - | \n \t`.

## 2. Chuẩn hóa tên cửa hàng (store_name)
- Áp dụng chuẩn hóa tiếng Việt chung.
- Loại bỏ các tiền tố nhận diện mang tính chung chung không thuộc tên thương hiệu nếu xuất hiện ở đầu chuỗi (không phân biệt chữ hoa thường):
  - "cửa hàng", "cua hang"
  - "siêu thị", "sieu thi"
  - Ví dụ: `"Cửa hàng Bách Hóa Xanh"` → `"Bách Hóa Xanh"`.

## 3. Chuẩn hóa địa chỉ (address)
- Áp dụng chuẩn hóa tiếng Việt chung.
- Loại bỏ tiền tố nhận diện địa chỉ nếu xuất hiện ở đầu chuỗi (không phân biệt chữ hoa thường):
  - "địa chỉ", "dia chi", "address"
  - Ví dụ: `"Địa chỉ: 123 Nguyễn Trãi, Q.5"` → `"123 Nguyễn Trãi, Q.5"`.

## 4. Chuẩn hóa ngày tháng (date)
- Chuẩn hóa toàn bộ về định dạng chuẩn duy nhất: **`YYYY-MM-DD`**.
- Loại bỏ các từ khóa ngày tháng giờ giấc như `"ngày", "ngay", "date", "time", "lúc", "luc", "giờ", "gio"`.
- Hỗ trợ phân tích (parse) các mẫu ngày tháng thông dụng sau bằng Regular Expression:
  - `dd/mm/yyyy` hoặc `dd-mm-yyyy` (ví dụ: `22/05/2024` → `2024-05-22`)
  - `dd/mm/yy` hoặc `dd-mm-yy` (ví dụ: `22/05/24` → `2024-05-22`, cộng thêm 2000 nếu năm < 100)
  - `yyyy/mm/dd` hoặc `yyyy-mm-dd` (ví dụ: `2024/05/22` → `2024-05-22`)
- Nếu chuỗi ngày tháng không hợp lệ hoặc không trùng khớp với bất kỳ mẫu nào ở trên, giá trị trả về bắt buộc phải là chuỗi rỗng `""`.

## 5. Chuẩn hóa số tiền (total)
- Chuẩn hóa về chuỗi ký tự chỉ chứa chữ số nguyên thuần túy (ví dụ: `"115000"`).
- Loại bỏ các từ khóa liên quan đến tiền tệ, thuế hay hóa đơn:
  - `"vnđ", "vnd", "đồng", "dong", "đ"`
  - `"tổng thanh toán", "tong thanh toan", "tổng cộng", "tong cong", "total", "amount"`
- Loại bỏ tất cả các ký tự phân tách hàng nghìn (dấu chấm `.`, dấu phẩy `,` hoặc khoảng trắng).
- Loại bỏ các số 0 ở đầu chuỗi không có nghĩa (leading zeros).
- Nếu không tìm thấy số tiền hợp lệ hoặc sau khi xử lý không chứa chữ số, giá trị trả về bắt buộc phải là chuỗi rỗng `""`.
- Ví dụ:
  - `"Tổng cộng: 115.000đ"` → `"115000"`
  - `"115,000 VND"` → `"115000"`
  - `"0115000"` → `"115000"`
  - `"Không đồng"` → `""`.
