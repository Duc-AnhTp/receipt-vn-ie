import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from typing import Tuple

def setup_donut_model_and_processor(
    model_name: str = "naver-clova-ix/donut-base",
    task_token: str = "<s_receipt_ie>",
    cord_task_token: str = "<s_cord_receipt_parse>"
) -> Tuple[VisionEncoderDecoderModel, DonutProcessor]:
    """
    Khởi tạo và cấu hình mô hình Donut cùng Processor.
    Thêm các special tokens mới vào tokenizer và cập nhật embedding size của decoder.
    """
    # 1. Load processor và model
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # 2. Định nghĩa các tokens đặc biệt mới cần thêm
    # Bao gồm các task tokens và các tag mở/đóng cho từng trường thông tin
    new_tokens = [
        task_token, 
        task_token.replace("<", "</"),
        cord_task_token,
        cord_task_token.replace("<", "</"),
        "<s_store_name>", "</s_store_name>",
        "<s_date>", "</s_date>",
        "<s_total>", "</s_total>",
        "<s_address>", "</s_address>"
    ]
    
    # Thêm tokens vào tokenizer
    # return_tensors="pt" hoặc chỉ thêm vào vocab
    num_added_tokens = processor.tokenizer.add_tokens(new_tokens)
    print(f"Đã thêm {num_added_tokens} tokens mới vào tokenizer.")
    
    # 3. Cập nhật model embeddings để nhận diện các token mới
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # 4. Cấu hình các tham số cho model
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Thiết lập decoder_start_token_id là task_token mặc định
    task_token_id = processor.tokenizer.convert_tokens_to_ids(task_token)
    model.config.decoder_start_token_id = task_token_id
    
    # Một số cấu hình chung cho generation của Donut
    model.config.vocab_size = len(processor.tokenizer)
    
    return model, processor
