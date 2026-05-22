import logging
from typing import Dict, List, Tuple, Any

# Cấu hình logging
logger = logging.getLogger(__name__)

from receipt_ie.data.schemas import LABELS, LABEL2ID, ID2LABEL

LABEL_TO_ID = LABEL2ID
ID_TO_LABEL = ID2LABEL

def compute_intersection_area(box1: List[int], box2: List[int]) -> int:
    """
    Tính diện tích giao nhau giữa hai bounding box dạng [x0, y0, x2, y2].
    """
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x0 < x2 and y0 < y2:
        return (x2 - x0) * (y2 - y0)
    return 0

def compute_box_area(box: List[int]) -> int:
    """
    Tính diện tích của bounding box dạng [x0, y0, x2, y2].
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width > 0 and height > 0:
        return width * height
    return 0

def assign_word_labels(
    words: List[str],
    word_boxes: List[List[int]],
    field_boxes: Dict[str, List[List[int]]],
    overlap_threshold: float = 0.5
) -> List[str]:
    """
    Gán nhãn BIO cho danh sách các word dựa trên độ giao nhau (overlap) giữa word_box
    và field_boxes của ground truth.
    
    field_boxes: dict dạng {"store_name": [[x0,y0,x2,y2], ...], ...}
    """
    word_labels = ["O"] * len(words)
    
    # Theo dõi trạng thái của từng field để gán B- hoặc I-
    # field_active: lưu nhãn field đang được xử lý ở từ trước đó
    prev_field = None
    
    for i, (word, w_box) in enumerate(zip(words, word_boxes)):
        w_area = compute_box_area(w_box)
        if w_area <= 0:
            word_labels[i] = "O"
            prev_field = None
            continue
            
        best_field = None
        best_overlap_ratio = 0.0
        
        # Tìm field có overlap lớn nhất với word box
        for field, f_boxes in field_boxes.items():
            if not f_boxes:
                continue
            for f_box in f_boxes:
                intersect = compute_intersection_area(w_box, f_box)
                ratio = intersect / w_area
                if ratio > best_overlap_ratio:
                    best_overlap_ratio = ratio
                    best_field = field
                    
        # Nếu overlap lớn hơn ngưỡng, gán nhãn
        if best_field and best_overlap_ratio >= overlap_threshold:
            # Xác định B- hay I-
            # Nếu từ liền trước cùng thuộc field này, gán I-
            # Ngược lại, gán B-
            if prev_field == best_field:
                word_labels[i] = f"I-{best_field.upper()}"
            else:
                word_labels[i] = f"B-{best_field.upper()}"
            prev_field = best_field
        else:
            word_labels[i] = "O"
            prev_field = None
            
    return word_labels

def normalize_bbox(box: List[int], width: int, height: int) -> List[int]:
    """
    Chuẩn hoá bounding box về dải [0, 1000] theo yêu cầu của LayoutLM.
    """
    if not box or len(box) < 4:
        return [0, 0, 0, 0]
        
    x0 = int(1000 * max(0, min(box[0], width)) / width)
    y0 = int(1000 * max(0, min(box[1], height)) / height)
    x2 = int(1000 * max(0, min(box[2], width)) / width)
    y2 = int(1000 * max(0, min(box[3], height)) / height)
    
    # Đảm bảo box hợp lệ x0 <= x2 và y0 <= y2
    x0, x2 = min(x0, x2), max(x0, x2)
    y0, y2 = min(y0, y2), max(y0, y2)
    
    return [x0, y0, x2, y2]

def align_tokens_layoutxlm(
    words: List[str],
    word_boxes: List[List[int]],
    word_labels: List[str],
    tokenizer: Any,
    max_length: int = 512,
    image_size: Tuple[int, int] = (1000, 1000)
) -> Dict[str, List[Any]]:
    """
    Thực hiện tokenize danh sách words và căn chỉnh nhãn BIO cùng bounding box 
    cho các token con (subwords) theo quy chuẩn LayoutXLM.
    
    - Subword đầu tiên nhận nhãn BIO của word và box chuẩn hoá.
    - Các subwords tiếp theo nhận nhãn -100 và box chuẩn hoá.
    - Token đặc biệt <s> nhận box [0,0,0,0] và nhãn -100.
    - Token đặc biệt </s> nhận box [1000,1000,1000,1000] và nhãn -100.
    - Padding nhận box [0,0,0,0] và nhãn -100.
    """
    img_w, img_h = image_size
    
    input_ids = []
    bbox = []
    labels = []
    
    # Token đầu tiên: <s> (BOS)
    input_ids.append(tokenizer.bos_token_id)
    bbox.append([0, 0, 0, 0])
    labels.append(-100)
    
    for word, box, label in zip(words, word_boxes, word_labels):
        # Chuẩn hoá box
        norm_box = normalize_bbox(box, img_w, img_h)
        
        # Tokenize word
        sub_tokens = tokenizer.tokenize(word)
        if not sub_tokens:
            continue
            
        sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        
        # Lấy ID của nhãn BIO
        label_id = LABEL_TO_ID.get(label, 0)
        
        # Subword đầu tiên
        input_ids.append(sub_ids[0])
        bbox.append(norm_box)
        labels.append(label_id)
        
        # Các subword tiếp theo nhận nhãn -100
        for sub_id in sub_ids[1:]:
            input_ids.append(sub_id)
            bbox.append(norm_box)
            labels.append(-100)
            
    # Token kết thúc: </s> (EOS)
    input_ids.append(tokenizer.eos_token_id)
    bbox.append([1000, 1000, 1000, 1000])
    labels.append(-100)
    
    # Xử lý Truncation (Cắt cụt)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length-1] + [tokenizer.eos_token_id]
        bbox = bbox[:max_length-1] + [[1000, 1000, 1000, 1000]]
        labels = labels[:max_length-1] + [-100]
        
    # Tạo attention_mask
    attention_mask = [1] * len(input_ids)
    
    # Xử lý Padding (Đệm)
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        bbox += [[0, 0, 0, 0]] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len
        
    return {
        "input_ids": input_ids,
        "bbox": bbox,
        "attention_mask": attention_mask,
        "labels": labels
    }
