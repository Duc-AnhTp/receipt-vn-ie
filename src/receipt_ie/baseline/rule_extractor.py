"""Rule-based field extraction from OCR cache."""
import re
import logging
from receipt_ie.data.normalize_text import (
    normalize_date
)
from receipt_ie.postprocess.store_extractor import extract_store_name_from_lines
from receipt_ie.postprocess.address_extractor import extract_address_from_lines
from receipt_ie.postprocess.total_extractor import extract_total_from_lines

logger = logging.getLogger(__name__)


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
    prediction["store_name"] = extract_store_name_from_lines(line_texts)

    # --- 2. TRÍCH XUẤT DATE ---
    # Duyệt từ trên xuống dưới tìm dòng khớp regex ngày tháng
    date_val = ""
    for text in line_texts:
        norm_d = normalize_date(text)
        if norm_d:
            date_val = norm_d
            break
    prediction["date"] = date_val

    # --- 3. TRÍCH XUẤT ADDRESS ---
    prediction["address"] = extract_address_from_lines(line_texts)

    # --- 4. TRÍCH XUẤT TOTAL ---
    prediction["total"] = extract_total_from_lines(line_texts)

    return prediction
