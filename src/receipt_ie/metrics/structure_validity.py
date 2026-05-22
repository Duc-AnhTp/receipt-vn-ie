from typing import Dict, List

def check_donut_structure(seq: str, task_token: str) -> Dict[str, bool]:
    """
    Kiểm tra tính đúng đắn cấu trúc của chuỗi XML sinh ra từ Donut.
    
    - task_tokens_valid: Cặp thẻ task (mở/đóng) xuất hiện đầy đủ ở đầu/cuối.
    - field_tags_valid: Mọi tag mở của các trường mục tiêu đều có tag đóng đi sau nó.
    """
    end_token = task_token.replace("<", "</")
    
    # 1. Kiểm tra cặp thẻ task
    has_task_start = seq.startswith(task_token) or task_token in seq
    has_task_end = seq.endswith(end_token) or end_token in seq
    task_tokens_valid = has_task_start and has_task_end
    
    # 2. Kiểm tra các tag trường mục tiêu
    fields = ["store_name", "date", "total", "address"]
    field_tags_valid = True
    
    for f in fields:
        start_tag = f"<s_{f}>"
        end_tag = f"</s_{f}>"
        
        start_count = seq.count(start_tag)
        end_count = seq.count(end_tag)
        
        # Nếu số lượng tag mở và tag đóng lệch nhau
        if start_count != end_count:
            field_tags_valid = False
            break
            
        # Kiểm tra xem tag đóng có luôn đi sau tag mở tương ứng không
        if start_count > 0:
            start_idx = 0
            for _ in range(start_count):
                start_idx = seq.find(start_tag, start_idx)
                end_idx = seq.find(end_tag, start_idx)
                if end_idx == -1 or end_idx < start_idx:
                    field_tags_valid = False
                    break
                start_idx += len(start_tag)
                
            if not field_tags_valid:
                break
                
    return {
        "task_tokens_valid": task_tokens_valid,
        "field_tags_valid": field_tags_valid,
        "overall_valid": task_tokens_valid and field_tags_valid
    }

def compute_structure_validity_rates(sequences: List[str], task_token: str) -> Dict[str, float]:
    """
    Tính toán tỷ lệ hợp lệ cấu trúc của danh sách các chuỗi sinh ra.
    """
    if not sequences:
        return {"task_tokens_valid_rate": 0.0, "field_tags_valid_rate": 0.0, "overall_valid_rate": 0.0}
        
    task_valid_count = 0
    field_valid_count = 0
    overall_valid_count = 0
    
    for seq in sequences:
        res = check_donut_structure(seq, task_token)
        if res["task_tokens_valid"]:
            task_valid_count += 1
        if res["field_tags_valid"]:
            field_valid_count += 1
        if res["overall_valid"]:
            overall_valid_count += 1
            
    n = len(sequences)
    return {
        "task_tokens_valid_rate": round(task_valid_count / n, 4),
        "field_tags_valid_rate": round(field_valid_count / n, 4),
        "overall_valid_rate": round(overall_valid_count / n, 4)
    }
