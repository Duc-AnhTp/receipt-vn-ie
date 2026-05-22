import torch
# Đảm bảo import torch đầu tiên trên Windows để tránh DLL collision
import time
from typing import Dict, Any
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from receipt_ie.data.schemas import BaseExtractor
from receipt_ie.data.build_donut_dataset import donut_sequence_to_target
from receipt_ie.inference.postprocess_json import postprocess_extracted_fields

class DonutExtractor(BaseExtractor):
    """
    Bộ trích xuất thông tin biên lai sử dụng mô hình Donut.
    Kế thừa interface BaseExtractor chung.
    """
    def __init__(self, task_token: str = "<s_receipt_ie>"):
        self.task_token = task_token
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, checkpoint_path: str) -> None:
        """
        Nạp mô hình Donut và Processor từ checkpoint.
        """
        print(f"Loading Donut model from {checkpoint_path}...")
        self.processor = DonutProcessor.from_pretrained(checkpoint_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print("Donut model loaded successfully.")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Thực hiện dự đoán ảnh biên lai.
        Trả về dictionary kết quả thô, kết quả chuẩn hoá và thời gian trích xuất (latency).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Mô hình chưa được nạp. Vui lòng gọi hàm load() trước.")
            
        start_time = time.time()
        
        # 1. Tiền xử lý ảnh qua processor
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # 2. Tạo decoder input bắt đầu bằng task token
        decoder_input_ids = self.processor.tokenizer(
            self.task_token, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # 3. Sinh chuỗi XML kết quả
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.config.max_length,
                num_beams=1,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
        # Giải mã kết quả
        seq = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        
        # 4. Parse chuỗi XML kết quả về dạng dict thô
        raw_pred = donut_sequence_to_target(seq, self.task_token)
        
        # 5. Chuẩn hoá kết quả
        norm_pred = postprocess_extracted_fields(raw_pred)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "prediction": raw_pred,
            "normalized_prediction": norm_pred,
            "raw_output": seq,
            "latency_ms": latency_ms
        }
