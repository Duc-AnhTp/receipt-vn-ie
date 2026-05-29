"""
Bộ trích xuất heuristics (Rule-based) từ kết quả OCR cache.
Sử dụng Regular Expressions và vị trí dòng text để trích xuất 4 trường thông tin.
"""
import re
import logging
from receipt_ie.data.normalize_text import (
    normalize_store_name,
    normalize_address,
    normalize_date,
    normalize_money
)

logger = logging.getLogger(__name__)


PHONE_OR_CODE_RE = re.compile(
    r"\b(tel|phone|đt|sđt|dt|sdt|fax|mst|tax|tax_code|mã số thuế|ma so thue|mã hđ|ma hd|invoice|bill no)\b",
    re.IGNORECASE,
)


TOTAL_KEYWORDS = [
    r"tổng cộng", r"tong cong", r"tổng tiền", r"tong tien", r"tổng thanh toán", r"tong thanh toan",
    r"thanh toán", r"thanh toan", r"total", r"amount", r"cần trả", r"can tra",
    r"thành tiền", r"thanh tien", r"phải trả", r"phai tra", r"tổng", r"tong"
]


def _money_candidates(text: str) -> list[str]:
    values = []
    for match in re.finditer(r"\d[\d\.,\s]*", text):
        raw = match.group(0)
        normalized = normalize_money(raw)
        if normalized and normalized.isdigit():
            values.append(normalized)
    return values


def _is_bad_money_context(text: str) -> bool:
    if normalize_date(text):
        return True
    if PHONE_OR_CODE_RE.search(text):
        return True
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 9 and not re.search("|".join(TOTAL_KEYWORDS), text, re.IGNORECASE):
        return True
    return False


def extract_fields_from_ocr(ocr_data: dict) -> dict:
    """
    Trích xuất 4 trường thông tin từ dữ liệu OCR cache của một hóa đơn.
    
    Args:
        ocr_data (dict): Dữ liệu OCR cache chứa các trường "lines" và "words".
        
    Returns:
        dict: Dự đoán 4 trường đã được chuẩn hóa:
            {
                "store_name": "...",
                "date": "...",
                "total": "...",
                "address": "..."
            }
    """
    lines = ocr_data.get("lines", [])
    
    # Ghép các từ trong mỗi dòng thành chuỗi text phẳng
    line_texts = []
    for line in lines:
        line_text = " ".join([word.get("text", "") for word in line if word.get("text")])
        line_text = re.sub(r"\s+", " ", line_text).strip()
        if line_text:
            line_texts.append(line_text)
            
    prediction = {
        "store_name": "",
        "date": "",
        "total": "",
        "address": ""
    }
    
    if not line_texts:
        return prediction

    # --- 1. TRÍCH XUẤT STORE_NAME ---
    # Duyệt tối đa 5 dòng đầu tiên
    candidate_store = ""
    for i, text in enumerate(line_texts[:5]):
        # Bỏ qua dòng chứa số điện thoại, MST, email, website
        if re.search(r"\b(tel|phone|đt|sđt|dt|sdt|fax|lh|mst|tax|tax_code|mã số thuế|email|website|web|www)\b", text, re.IGNORECASE):
            continue
        # Bỏ qua dòng chứa quá nhiều chữ số (như mã hóa đơn, ngày tháng)
        digit_ratio = sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0
        if digit_ratio > 0.4:
            continue
        # Bỏ qua dòng chỉ chứa ký tự phân cách
        clean_text = re.sub(r"[_\-*=+\|/\\]", "", text).strip()
        if len(clean_text) < 3:
            continue
            
        candidate_store = text
        break
        
    if not candidate_store and line_texts:
        candidate_store = line_texts[0]
        
    prediction["store_name"] = normalize_store_name(candidate_store)

    # --- 2. TRÍCH XUẤT DATE ---
    date_val = ""
    date_candidates = []
    for idx, text in enumerate(line_texts):
        norm_d = normalize_date(text)
        if norm_d:
            score = 0 if re.search(r"\b(ngày|ngay|date|time|giờ|gio|bán hàng|ban hang)\b", text, re.IGNORECASE) else 1
            date_candidates.append((score, idx, norm_d))
    if date_candidates:
        date_val = sorted(date_candidates, key=lambda x: (x[0], x[1]))[0][2]
    prediction["date"] = date_val

    # --- 3. TRÍCH XUẤT ADDRESS ---
    # Tìm tất cả các dòng chứa từ khóa hành chính Việt Nam hoặc địa chỉ
    address_keywords = [
        r"\bđường\b", r"\bduong\b", r"\bphố\b", r"\bpho\b", r"\bngõ\b", r"\bngo\b",
        r"\bquận\b", r"\bquan\b", r"\bhuyện\b", r"\bhuyen\b", r"\bthành phố\b", r"\bthanh pho\b",
        r"\btp\b", r"\btỉnh\b", r"\btinh\b", r"\bphường\b", r"\bphuong\b", r"\bxã\b", r"\bxa\b",
        r"\bấp\b", r"\bap\b", r"\bthôn\b", r"\bthon\b", r"\btổ\b", r"\bto\b", r"\bkhu\b",
        r"\bđịa chỉ\b", r"\bdia chi\b", r"\baddress\b"
    ]
    address_pattern = re.compile("|".join(address_keywords), re.IGNORECASE)
    
    address_lines = []
    for idx, text in enumerate(line_texts):
        # Bỏ qua các dòng chứa từ khóa tổng tiền để tránh nhầm lẫn địa chỉ với tổng tiền ở cuối
        if re.search(r"\b(tổng cộng|tong cong|tổng tiền|tong tien|thanh toán|thanh toan|total|amount|cần trả|can tra|thành tiền|thanh tien)\b", text, re.IGNORECASE):
            continue
            
        if address_pattern.search(text):
            address_lines.append(text)
            # Thử lấy thêm dòng tiếp theo nếu nó không chứa các từ khóa cấm và không quá ngắn
            if idx + 1 < len(line_texts):
                next_text = line_texts[idx + 1]
                if (len(next_text) > 5 and 
                    not address_pattern.search(next_text) and
                    not re.search(r"\b(tel|phone|đt|sđt|fax|mst|tax|total|tổng cộng|ngày|ngay)\b", next_text, re.IGNORECASE)):
                    address_lines.append(next_text)
            break  # Lấy cụm địa chỉ đầu tiên tìm thấy
            
    prediction["address"] = normalize_address(" ".join(address_lines))

    # --- 4. TRÍCH XUẤT TOTAL ---
    # Duyệt từ dưới lên trên tìm dòng chứa từ khóa tổng cộng
    total_pattern = re.compile(r"\b(" + "|".join(TOTAL_KEYWORDS) + r")\b", re.IGNORECASE)
    
    total_val = ""
    found_total = False
    
    for idx in range(len(line_texts) - 1, -1, -1):
        text = line_texts[idx]
        if total_pattern.search(text):
            window = [text]
            if idx + 1 < len(line_texts):
                window.append(line_texts[idx + 1])
            if idx - 1 >= 0:
                window.append(line_texts[idx - 1])

            candidates = []
            for candidate_text in window:
                if _is_bad_money_context(candidate_text):
                    continue
                for money in _money_candidates(candidate_text):
                    value = int(money)
                    if value > 0:
                        candidates.append(money)
            if candidates:
                total_val = max(candidates, key=lambda x: int(x))
                found_total = True
                break

    # Fallback: Nếu không tìm thấy qua từ khóa, tìm cụm số lớn nhất ở nửa cuối hóa đơn
    if not found_total:
        candidates = []
        start_idx = int(len(line_texts) * 0.66)
        for text in line_texts[start_idx:]:
            if _is_bad_money_context(text):
                continue
            for norm_money in _money_candidates(text):
                val = int(norm_money)
                if 1000 <= val <= 1000000000:
                    candidates.append(norm_money)
        if candidates:
            # Ưu tiên số tiền lớn nhất (thường là tổng tiền so với các đơn giá lẻ)
            total_val = max(candidates, key=lambda x: int(x))
            
    prediction["total"] = total_val

    return prediction
