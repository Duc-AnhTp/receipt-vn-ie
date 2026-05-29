import time
from typing import Dict, Any
from PIL import Image

from receipt_ie.data.schemas import BaseExtractor

class MockExtractor(BaseExtractor):
    """
    Mock Extractor fallback khi không tìm thấy checkpoint thực tế của Donut hoặc LayoutXLM.
    Sử dụng BaselineExtractor làm backend và mô phỏng kết quả của Donut/LayoutXLM.
    """
    def __init__(self, method: str, baseline_extractor):
        self.method = method.lower()
        self.baseline = baseline_extractor
        
    def load(self, checkpoint_path: str) -> None:
        pass
        
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        # Chạy baseline để có thông tin thật trên ảnh
        res = self.baseline.predict(image)
        pred = res["prediction"]
        norm_pred = res["normalized_prediction"]
        words = res.get("words", [])
        
        if self.method == "donut":
            # Tạo chuỗi tag-sequence thô giả lập
            seq = "<s_receipt_ie>"
            if pred.get("store_name"):
                seq += f"<s_store_name>{pred['store_name']}</s_store_name>"
            if pred.get("date"):
                seq += f"<s_date>{pred['date']}</s_date>"
            if pred.get("total"):
                seq += f"<s_total>{pred['total']}</s_total>"
            if pred.get("address"):
                seq += f"<s_address>{pred['address']}</s_address>"
            seq += "</s_receipt_ie>"
            
            return {
                "prediction": pred,
                "normalized_prediction": norm_pred,
                "raw_output": seq,
                "latency_ocr_ms": res.get("latency_ocr_ms", 0.0) * 1.1,
                "latency_model_ms": res.get("latency_model_ms", res.get("latency_cached_ms", 0.0)) * 1.1,
                "latency_postprocess_ms": res.get("latency_postprocess_ms", 0.0) * 1.1,
                "latency_cached_ms": res["latency_cached_ms"] * 1.1, # giả lập chậm hơn xíu
                "latency_e2e_ms": res["latency_e2e_ms"] * 1.1,
                "status": "ok",
                "is_mock": True
            }
        else: # layoutxlm
            # Giả lập nhãn BIO cho từng word
            word_labels = []
            for w in words:
                text = w["text"].lower()
                # Thử tìm xem word nằm ở trường nào
                assigned = False
                for field in ["store_name", "date", "total", "address"]:
                    val = pred.get(field, "").lower()
                    if val and text in val:
                        # Giả lập nhãn BIO đơn giản
                        prefix = "B-" if not any(l.endswith(field.upper()) for l in word_labels[-1:]) else "I-"
                        word_labels.append(f"{prefix}{field.upper()}")
                        assigned = True
                        break
                if not assigned:
                    word_labels.append("O")
                    
            return {
                "prediction": pred,
                "normalized_prediction": norm_pred,
                "latency_ocr_ms": res.get("latency_ocr_ms", 0.0) * 1.2,
                "latency_model_ms": res.get("latency_model_ms", res.get("latency_cached_ms", 0.0)) * 1.2,
                "latency_postprocess_ms": res.get("latency_postprocess_ms", 0.0) * 1.2,
                "latency_cached_ms": res["latency_cached_ms"] * 1.2,
                "latency_e2e_ms": res["latency_e2e_ms"] * 1.2,
                "status": "ok",
                "words": words,
                "word_labels": word_labels,
                "is_mock": True
            }
