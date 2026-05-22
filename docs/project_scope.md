# Phạm vi đề tài (Project Scope)

Tài liệu này xác định rõ ràng mục tiêu, dữ liệu đầu vào, kết quả đầu ra và phạm vi hoạt động của hệ thống trích xuất thông tin biên lai tiếng Việt.

## 1. Dữ liệu đầu vào (Input)
- Ảnh chụp biên lai thanh toán bằng tiếng Việt (định dạng JPG, PNG).
- Chấp nhận ảnh chụp từ điện thoại di động, có thể có hiện tượng nghiêng, mờ nhẹ, thiếu sáng hoặc nhàu nát ở mức độ vừa phải.

## 2. Tiêu chuẩn đầu ra (Output Schema)
Kết quả trích xuất của hệ thống bắt buộc phải được tổ chức thành cấu trúc JSON với 4 trường thông tin sau:
- **store_name**: Tên cửa hàng, siêu thị hoặc thương hiệu đơn vị bán hàng.
- **date**: Ngày in hóa đơn hoặc ngày thực hiện giao dịch (định dạng chuẩn hóa `YYYY-MM-DD`).
- **total**: Tổng số tiền thực tế khách hàng phải thanh toán (chuỗi chỉ chứa chữ số nguyên, ví dụ: `"115000"`).
- **address**: Địa chỉ của cửa hàng nơi giao dịch diễn ra.

*Ví dụ đầu ra mẫu:*
```json
{
  "store_name": "MINIMART ANAN",
  "date": "2020-08-09",
  "total": "115000",
  "address": "Chợ Sủi Phú Thị, Gia Lâm"
}
```

## 3. Phạm vi không thực hiện (Out of Scope)
Để tập trung tối đa vào mục tiêu so sánh hiệu quả của hai kiến trúc Donut và LayoutXLM, hệ thống **không** thực hiện các chức năng sau:
- Không trích xuất danh sách chi tiết các sản phẩm (items list).
- Không trích xuất thuế VAT, mã số thuế (MST) của đơn vị bán hàng (trừ khi MST trùng làm tên cửa hàng).
- Không trích xuất số điện thoại, email, website hoặc hotline.
- Không trích xuất mã hóa đơn, hình thức thanh toán hoặc tiền thừa.
- Không xử lý các file tài liệu dạng PDF số hóa, chỉ làm việc trực tiếp trên dữ liệu dạng ảnh chụp biên lai giấy.
- Không huấn luyện mô hình phân loại chất lượng ảnh của MC-OCR.

## 4. Thiết lập tập dữ liệu thử nghiệm chính (Main Test Set)
- Hệ thống chỉ được đánh giá và so sánh metrics trên tập test tiếng Việt được trích xuất từ tập **MC-OCR 2021 có nhãn** kết hợp với **dữ liệu tự thu thập**.
- Tập dữ liệu CORD v2 (ngôn ngữ Tiếng Anh/Tiếng Indonesia) không được đưa vào tập kiểm tra chính để đảm bảo kết quả phản ánh chính xác năng lực xử lý ngôn ngữ tiếng Việt của các mô hình.
