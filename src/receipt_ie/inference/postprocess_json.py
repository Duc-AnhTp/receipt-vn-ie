from typing import Dict
from receipt_ie.data.normalize_text import (
    normalize_store_name,
    normalize_date,
    normalize_money,
    normalize_address
)

def postprocess_extracted_fields(raw_pred: Dict[str, str]) -> Dict[str, str]:
    """
    Chuẩn hoá và làm sạch các trường thông tin trích xuất thô theo các quy tắc
    được định nghĩa trong normalization_rules.md.
    """
    return {
        "store_name": normalize_store_name(raw_pred.get("store_name", "")),
        "date": normalize_date(raw_pred.get("date", "")),
        "total": normalize_money(raw_pred.get("total", "")),
        "address": normalize_address(raw_pred.get("address", ""))
    }
