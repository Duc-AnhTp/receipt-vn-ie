import re
import unicodedata
from datetime import datetime

def normalize_vietnamese_text(s: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt về Unicode NFC, loại bỏ khoảng trắng thừa
    và các ký tự đặc biệt ở đầu/cuối chuỗi.
    """
    if s is None:
        return ""
    
    s = str(s)
    # Đưa về chuẩn dựng sẵn Unicode NFC
    s = unicodedata.normalize("NFC", s)
    # Thay thế dấu cách cứng (non-breaking space)
    s = s.replace("\u00a0", " ")
    # Gộp nhiều khoảng trắng liên tiếp thành 1 khoảng trắng
    s = re.sub(r"\s+", " ", s)
    # Loại bỏ các ký tự đặc biệt thừa ở đầu/cuối
    s = s.strip(" :;-|\n\t")
    
    return s

def normalize_store_name(s: str) -> str:
    """
    Chuẩn hóa tên cửa hàng, loại bỏ các tiền tố 'cửa hàng', 'siêu thị'...
    """
    s = normalize_vietnamese_text(s)
    # Loại bỏ tiền tố không cần thiết ở đầu chuỗi
    s = re.sub(r"^(cửa hàng|cua hang|siêu thị|sieu thi)\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def normalize_address(s: str) -> str:
    """
    Chuẩn hóa địa chỉ, loại bỏ các tiền tố 'địa chỉ', 'address'...
    """
    s = normalize_vietnamese_text(s)
    # Loại bỏ tiền tố không cần thiết ở đầu chuỗi
    s = re.sub(r"^(địa chỉ|dia chi|address)\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def normalize_date(s: str) -> str:
    """
    Chuẩn hóa các định dạng ngày tháng tiếng Việt về định dạng chuẩn YYYY-MM-DD.
    Nếu không parse được hoặc không khớp mẫu, trả về chuỗi rỗng.
    """
    raw = normalize_vietnamese_text(s).lower()
    keyword_re = re.compile(r"(ngày|ngay|date|time|lúc|luc|giờ|gio|bán hàng|ban hang)")

    # Các mẫu regex ngày tháng thông dụng. Hỗ trợ /, -, . và năm 2 chữ số.
    patterns = [
        (r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b", "ymd"),
        (r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", "dmy"),
        (r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2})\b", "dmy"),
    ]

    candidates = []
    for pattern, order in patterns:
        for match in re.finditer(pattern, raw):
            parts = match.groups()
            candidates.append((0 if keyword_re.search(raw[max(0, match.start() - 24): match.end() + 24]) else 1, match.start(), parts, order))

    for _, _, parts, order in sorted(candidates, key=lambda x: (x[0], x[1])):
        try:
            if order == "ymd":
                year, month, day = map(int, parts)
            else:
                day, month, year = map(int, parts)
                if year < 100:
                    year += 2000

            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return ""

def normalize_money(s: str) -> str:
    """
    Chuẩn hóa chuỗi số tiền về dạng chuỗi chỉ chứa chữ số nguyên.
    Nếu không hợp lệ, trả về chuỗi rỗng.
    """
    s = normalize_vietnamese_text(s).lower()
    
    # Loại bỏ các đơn vị tiền tệ và các từ khóa liên quan
    s = re.sub(r"(vnđ|vnd|đồng|dong|đ)", "", s)
    s = re.sub(r"(tổng thanh toán|tong thanh toan|tổng cộng|tong cong|total|amount)", "", s)
    
    # Lấy các cụm ký tự số và các dấu phân tách
    nums = re.findall(r"\d[\d\.,\s]*", s)
    if not nums:
        return ""
    
    # Chọn cụm số có độ dài lớn nhất (tránh lấy các số phụ nhỏ lẻ khác)
    num = max(nums, key=len)
    
    # Loại bỏ tất cả ký tự không phải chữ số (dấu chấm, dấu phẩy, khoảng trắng...)
    num = re.sub(r"[^\d]", "", num)
    
    # Loại bỏ số 0 vô nghĩa ở đầu
    num = num.lstrip("0")
    if not num:
        return "0" if "0" in s else ""
        
    return num
