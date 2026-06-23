import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from typing import Tuple

def setup_donut_model_and_processor(
    model_name: str = "naver-clova-ix/donut-base",
    task_token: str = "<s_receipt_ie>",
    image_size: Tuple[int, int] = None
) -> Tuple[VisionEncoderDecoderModel, DonutProcessor]:
    """
    Khởi tạo và cấu hình mô hình Donut cùng Processor.
    Thêm các special tokens mới vào tokenizer và cập nhật embedding size của decoder.
    """
    # 1. Load processor và model
    processor = DonutProcessor.from_pretrained(model_name)
    if image_size is not None:
        processor.image_processor.size = {"height": image_size[0], "width": image_size[1]}
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # 2. Định nghĩa các tokens đặc biệt mới cần thêm
    # Bao gồm các task tokens và các tag mở/đóng cho từng trường thông tin
    new_tokens = [
        task_token, 
        task_token.replace("<", "</"),
        "<s_store_name>", "</s_store_name>",
        "<s_date>", "</s_date>",
        "<s_total>", "</s_total>",
        "<s_address>", "</s_address>"
    ]
    
    # Thêm tokens vào tokenizer dưới dạng special tokens để bảo vệ cấu trúc thẻ
    num_added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {num_added_tokens} new special tokens to tokenizer.")
    
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
