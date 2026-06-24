import re
import unicodedata
from datetime import datetime

VALID_YEAR_RANGE = (1990, 2040)
PHONE_RE = re.compile(r"^0[3-9]\d{8}$")
TAX_RE = re.compile(r"^\d{10}(\d{3})?$")
TOTAL_KEYWORD_RE = re.compile(
    r"(tổng thanh toán|tong thanh toan|tổng cộng|tong cong|thanh toán|"
    r"thanh toan|total|amount|phải trả|phai tra|thành tiền|thanh tien)",
    flags=re.IGNORECASE,
)

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
    s = normalize_vietnamese_text(s).lower()
    
    # Loại bỏ các từ khóa ngày tháng giờ giấc
    s = re.sub(r"(ngày|ngay|date|time|lúc|luc|giờ|gio)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    
    # Các mẫu regex ngày tháng thông dụng (sắp xếp từ cụ thể/dài nhất đến ngắn nhất)
    # Hỗ trợ dấu chấm (.) làm dấu phân cách
    patterns = [
        r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b",  # yyyy/mm/dd hoặc yyyy-mm-dd hoặc yyyy.mm.dd
        r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b",  # dd/mm/yyyy hoặc dd-mm-yyyy hoặc dd.mm.yyyy
        r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b",    # dd/mm/yy hoặc dd-mm-yy hoặc dd.mm.yy
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s)
        if not match:
            continue
        
        parts = match.groups()
        try:
            if len(parts[0]) == 4: # yyyy/mm/dd
                year, month, day = map(int, parts)
            else: # dd/mm/yyyy hoặc dd/mm/yy
                day, month, year = map(int, parts)
                if year < 100: # yy -> yyyy
                    year += 2000
            
            # Giới hạn dải năm giao dịch hợp lệ
            if VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]:
                # Trả về chuỗi ngày chuẩn hóa nếu ngày tháng năm hợp lệ
                return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            # Nếu giá trị ngày tháng không thực tế (ví dụ ngày 32, tháng 13)
            continue
            
    return ""

def _is_noise_number(num: str) -> bool:
    return bool(PHONE_RE.fullmatch(num) or TAX_RE.fullmatch(num))


def _canonical_money_digits(raw_digits: str) -> str:
    stripped = raw_digits.lstrip("0")
    return stripped if stripped else "0"


def normalize_money(s: str) -> str:
    """
    Chuẩn hóa chuỗi số tiền về dạng chuỗi chỉ chứa chữ số nguyên.
    Nếu không hợp lệ, trả về chuỗi rỗng.
    """
    s = normalize_vietnamese_text(s).lower()
    has_total_keyword = bool(TOTAL_KEYWORD_RE.search(s))

    s = re.sub(r"(vnđ|vnd|đồng|dong|đ)", "", s)
    s = TOTAL_KEYWORD_RE.sub("", s)

    nums = re.findall(r"\d[\d\.,\s]*", s)
    candidates: list[tuple[str, str]] = []

    for raw_num in nums:
        raw_digits = re.sub(r"[^\d]", "", raw_num)
        if not raw_digits:
            continue

        normalized_digits = _canonical_money_digits(raw_digits)
        candidates.append((raw_digits, normalized_digits))

    if not candidates:
        return ""

    non_noise = [
        normalized
        for raw_digits, normalized in candidates
        if not _is_noise_number(raw_digits)
    ]

    if non_noise:
        valid_candidates = non_noise
    elif has_total_keyword:
        valid_candidates = [normalized for _, normalized in candidates]
    else:
        return ""

    return max(valid_candidates, key=lambda num: (len(num), int(num)))
