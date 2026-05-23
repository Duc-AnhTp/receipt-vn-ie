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
            "method": "donut",
            "latency_cached_ms": latency_ms,
            "latency_e2e_ms": latency_ms,
            "status": "ok",
            "error": None
        }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Chạy suy luận Donut trên tập test.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/donut/receipt_ie/finetune/best_model",
        help="Đường dẫn đến checkpoint tốt nhất của Donut"
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        default="data/processed/test.jsonl",
        help="Đường dẫn file test.jsonl"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="outputs/predictions/donut_test.jsonl",
        help="Đường dẫn file JSONL đầu ra"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số mẫu chạy"
    )
    return parser.parse_args()


def main():
    import json
    import os
    from pathlib import Path
    from tqdm import tqdm
    
    args = parse_args()
    
    test_file = Path(args.test_jsonl)
    if not test_file.exists():
        print(f"Không tìm thấy file test: {test_file}")
        return
        
    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    extractor = DonutExtractor()
    extractor.load(args.checkpoint)
    
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
                
    if args.limit is not None:
        samples = samples[:args.limit]
        
    print(f"Running Donut inference on {len(samples)} samples...")
    count = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="Donut Inference"):
            sample_id = sample.get("id")
            image_path_str = sample.get("image_path")
            
            prediction_record = {
                "id": sample_id,
                "method": "donut",
                "prediction": {},
                "normalized_prediction": {},
                "raw_output": None,
                "latency_cached_ms": 0.0,
                "latency_e2e_ms": 0.0,
                "status": "ok",
                "error": None
            }
            
            try:
                if not image_path_str or not os.path.exists(image_path_str):
                    raise FileNotFoundError(f"Image not found at {image_path_str}")
                    
                img = Image.open(image_path_str).convert("RGB")
                res = extractor.predict(img)
                
                prediction_record["prediction"] = res["prediction"]
                prediction_record["normalized_prediction"] = res["normalized_prediction"]
                prediction_record["raw_output"] = res["raw_output"]
                prediction_record["latency_cached_ms"] = round(res["latency_cached_ms"], 2)
                prediction_record["latency_e2e_ms"] = round(res["latency_e2e_ms"], 2)
                
            except Exception as e:
                print(f"Error evaluating sample {sample_id}: {e}")
                prediction_record["status"] = "error"
                prediction_record["error"] = str(e)
                
            out_f.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Donut inference completed. Saved {count} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
