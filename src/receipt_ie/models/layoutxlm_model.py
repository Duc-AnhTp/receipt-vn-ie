import torch
from transformers import AutoTokenizer, LayoutLMv2ForTokenClassification
from typing import Tuple, Dict

from receipt_ie.data.schemas import LABEL2ID, ID2LABEL

def setup_layoutxlm_model_and_tokenizer(
    model_name: str = "microsoft/layoutxlm-base"
) -> Tuple[LayoutLMv2ForTokenClassification, AutoTokenizer]:
    """
    Khởi tạo và cấu hình mô hình LayoutXLM cùng Tokenizer cho tác vụ Token Classification.
    Sử dụng ánh xạ nhãn từ schemas.py.
    """
    # 1. Khởi tạo Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Khởi tạo Model LayoutLMv2ForTokenClassification
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL
    )
    
    return model, tokenizer
