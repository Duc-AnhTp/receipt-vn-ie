import re
import json
from typing import Dict, Any

def target_to_donut_sequence(target: Dict[str, Any], task_token: str) -> str:
    """
    Chuyển đổi một dict target sang chuỗi sequence dạng XML của Donut.
    Ví dụ:
    {
        "store_name": "MINIMART ANAN",
        "date": "2020-08-09",
        "total": "115000",
        "address": "Chợ Sủi Phú Thị Gia Lâm"
    }
    ->
    "<s_receipt_ie><s_store_name>MINIMART ANAN</s_store_name><s_date>2020-08-09</s_date><s_total>115000</s_total><s_address>Chợ Sủi Phú Thị Gia Lâm</s_address></s_receipt_ie>"
    """
    sequence = task_token
    
    # Nếu là task CORD warm-up, chỉ lấy total
    if "cord" in task_token:
        total = target.get("total", "")
        if total:
            sequence += f"<s_total>{total}</s_total>"
    else:
        # Lấy theo thứ tự store_name, date, total, address
        for field in ["store_name", "date", "total", "address"]:
            val = target.get(field, "")
            # Đảm bảo không ghi đè giá trị None thành chuỗi "None"
            val_str = str(val) if val is not None else ""
            sequence += f"<s_{field}>{val_str}</s_{field}>"
            
    sequence += task_token.replace("<", "</")
    return sequence

def donut_sequence_to_target(sequence: str, task_token: str) -> Dict[str, str]:
    """
    Parse chuỗi XML của Donut sinh ra về lại dạng dict.
    Hỗ trợ cả việc parse lỗi khi mô hình sinh ra tag thiếu/sai cấu trúc (sử dụng regex fallback).
    """
    target = {
        "store_name": "",
        "date": "",
        "total": "",
        "address": ""
    }
    
    # Loại bỏ task tokens ở đầu và cuối nếu có
    clean_seq = sequence
    clean_seq = clean_seq.replace(task_token, "")
    end_token = task_token.replace("<", "</")
    clean_seq = clean_seq.replace(end_token, "")
    
    # Dùng regex để tìm các cặp thẻ <s_field>value</s_field>
    # Regex này hỗ trợ bắt cả trường hợp tag kết thúc thiếu dấu gạch chéo hoặc sai tên tag nhẹ
    fields = ["store_name", "date", "total", "address"]
    for field in fields:
        pattern = rf"<s_{field}>(.*?)</s_{field}>"
        match = re.search(pattern, clean_seq, re.DOTALL)
        if match:
            target[field] = match.group(1).strip()
        else:
            # Fallback nếu không có tag đóng chính xác (ví dụ mô hình bị cắt cụt)
            fallback_pattern = rf"<s_{field}>(.*?)($|<s_)"
            match_fb = re.search(fallback_pattern, clean_seq, re.DOTALL)
            if match_fb:
                val = match_fb.group(1).strip()
                # Loại bỏ các tag thừa nếu có lọt vào
                val = re.sub(r"<.*?>", "", val)
                target[field] = val
                
    return target
