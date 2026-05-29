import torch
# Đảm bảo import torch đầu tiên trên Windows để tránh DLL collision
import time
import yaml
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
from transformers import AutoTokenizer, LayoutLMv2ForTokenClassification

from receipt_ie.data.schemas import BaseExtractor, ID2LABEL
from receipt_ie.ocr.detect_paddle import load_paddle_detector, detect_text_regions, crop_region
from receipt_ie.ocr.recognize_vietocr import load_vietocr_model, recognize_regions
from receipt_ie.ocr.reading_order import sort_reading_order
from receipt_ie.data.build_layoutxlm_labels import normalize_bbox
from receipt_ie.inference.postprocess_json import postprocess_extracted_fields
from receipt_ie.ocr.preprocess import rectify_document

EMPTY_FIELDS = {"store_name": "", "date": "", "total": "", "address": ""}


def _empty_fields() -> Dict[str, str]:
    return EMPTY_FIELDS.copy()


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except (PermissionError, OSError):
        pass


class LayoutXLMExtractor(BaseExtractor):
    """
    Bộ trích xuất thông tin biên lai sử dụng mô hình LayoutXLM (OCR-based).
    Kế thừa interface BaseExtractor chung.
    """
    def __init__(
        self,
        detector=None,
        recognizer=None,
        ocr_config_path: str = "configs/ocr.yaml",
        project_root: str = "."
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.ocr_config_path = Path(project_root) / ocr_config_path
        self.project_root = Path(project_root)
        
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.y_threshold = 12

    def load(self, checkpoint_path: str) -> None:
        """
        Nạp mô hình LayoutXLM và Tokenizer từ checkpoint.
        Đồng thời khởi tạo luôn bộ máy OCR nếu chưa có.
        """
        print(f"Loading LayoutXLM model from {checkpoint_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Nạp Image Processor cho LayoutXLM
        from transformers import LayoutLMv2ImageProcessor
        try:
            self.image_processor = LayoutLMv2ImageProcessor.from_pretrained(checkpoint_path)
        except Exception:
            self.image_processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutxlm-base")
            
        print("LayoutXLM model loaded successfully.")
        
        # Tự động nạp bộ OCR nếu chưa được truyền từ ngoài
        if self.detector is None or self.recognizer is None:
            self._init_ocr()

    def _init_ocr(self):
        """
        Đọc file cấu hình ocr.yaml và khởi tạo PaddleOCR + VietOCR.
        """
        if not self.ocr_config_path.exists():
            print(f"Cảnh báo: Không tìm thấy file config OCR tại {self.ocr_config_path}. Dùng cấu hình mặc định CPU.")
            use_gpu = False
            rec_config = "vgg_transformer"
        else:
            with open(self.ocr_config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            use_gpu = cfg.get("detection", {}).get("gpu", False)
            rec_config = cfg.get("recognition", {}).get("default_config", "vgg_transformer")
            self.y_threshold = cfg.get("cache", {}).get("reading_order_y_threshold", 12)
            
        print("Initializing OCR engines for LayoutXLM inference...")
        if self.detector is None:
            self.detector = load_paddle_detector(use_gpu=use_gpu, use_angle_cls=False, lang="vi")
        if self.recognizer is None:
            # Nhận dạng dùng GPU nếu có thể
            rec_gpu = torch.cuda.is_available()
            self.recognizer = load_vietocr_model(config_name=rec_config, use_gpu=rec_gpu)
        print("OCR engines initialized.")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Thực hiện suy luận qua pipeline: OCR -> Sắp xếp thứ tự đọc -> LayoutXLM dự đoán -> Ghép thực thể.
        Đo lường thời gian suy luận e2e (bao gồm OCR) và thời gian mô hình (cached).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Mô hình chưa được nạp. Vui lòng gọi hàm load() trước.")
            
        start_e2e = time.time()
        
        # Tự động căn thẳng ảnh hóa đơn nếu có viền nghiêng
        image = rectify_document(image)
        
        width, height = image.size
        
        # 1. Chạy OCR (PaddleOCR + VietOCR)
        # Vì hàm detect_text_regions nhận đường dẫn ảnh, ta lưu tạm ảnh ra đĩa
        # Lưu vào thư mục tạm trong workspace
        temp_dir = self.project_root / "data/temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_img_path = temp_dir / f"layoutxlm_{uuid.uuid4().hex}.png"
        image.save(temp_img_path)
        
        try:
            regions = detect_text_regions(self.detector, str(temp_img_path))
            
            if not regions:
                return {
                    "method": "layoutxlm",
                    "prediction": _empty_fields(),
                    "normalized_prediction": _empty_fields(),
                    "raw_output": None,
                    "latency_cached_ms": 0.0,
                    "latency_e2e_ms": (time.time() - start_e2e) * 1000,
                    "status": "ok",
                    "error": None,
                    "words": [],
                    "word_labels": []
                }
                
            cropped_imgs = [crop_region(image, r["bbox"], padding=2) for r in regions]
            # Sử dụng batch size 16 để recognizer chạy nhanh
            texts = recognize_regions(self.recognizer, cropped_imgs, batch_size=16)
            
            for r, text in zip(regions, texts):
                r["text"] = text.strip()
                
            # Lọc bỏ text rỗng
            regions = [r for r in regions if r["text"]]
        except Exception as exc:
            return {
                "method": "layoutxlm",
                "prediction": _empty_fields(),
                "normalized_prediction": _empty_fields(),
                "raw_output": None,
                "latency_cached_ms": 0.0,
                "latency_e2e_ms": (time.time() - start_e2e) * 1000,
                "status": "error",
                "error": str(exc),
                "words": [],
                "word_labels": []
            }
        finally:
            _cleanup_temp_file(temp_img_path)
            
        if not regions:
            return {
                "method": "layoutxlm",
                "prediction": _empty_fields(),
                "normalized_prediction": _empty_fields(),
                "raw_output": None,
                "latency_cached_ms": 0.0,
                "latency_e2e_ms": (time.time() - start_e2e) * 1000,
                "status": "ok",
                "error": None,
                "words": [],
                "word_labels": []
            }
            
        # 2. Sắp xếp thứ tự đọc tự nhiên
        flat_words, _ = sort_reading_order(regions, y_threshold=self.y_threshold)
        words = [w["text"] for w in flat_words]
        word_boxes = [w["bbox"] for w in flat_words]
        
        start_model = time.time()
        
        # 3. Chuẩn bị token và bbox cho LayoutXLM
        input_ids = []
        bbox = []
        
        # Token bắt đầu: <s>
        input_ids.append(self.tokenizer.bos_token_id)
        bbox.append([0, 0, 0, 0])
        
        # Phân rã words thành subwords và theo vết token con đầu tiên
        word_subword_lengths = []
        for word, box in zip(words, word_boxes):
            norm_box = normalize_bbox(box, width, height)
            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens:
                word_subword_lengths.append(0)
                continue
                
            sub_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            word_subword_lengths.append(len(sub_ids))
            
            for sub_id in sub_ids:
                input_ids.append(sub_id)
                bbox.append(norm_box)
                
        # Token kết thúc: </s>
        input_ids.append(self.tokenizer.eos_token_id)
        bbox.append([1000, 1000, 1000, 1000])
        
        # Cắt cụt nếu vượt quá max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length-1] + [self.tokenizer.eos_token_id]
            bbox = bbox[:self.max_length-1] + [[1000, 1000, 1000, 1000]]
            
        # Tạo attention mask
        attention_mask = [1] * len(input_ids)
        
        # Nạp dữ liệu vào PyTorch Tensor
        tensor_input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        tensor_bbox = torch.tensor([bbox], dtype=torch.long).to(self.device)
        tensor_attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # Tiền xử lý ảnh cho model (chuyển đổi sang BGR 224x224 và chuẩn hóa)
        tensor_image = self.image_processor(image, apply_ocr=False, return_tensors="pt").pixel_values.to(self.device)
        
        # 4. Chạy mô hình LayoutXLM dự đoán
        with torch.no_grad():
            outputs = self.model(
                input_ids=tensor_input_ids,
                bbox=tensor_bbox,
                image=tensor_image,
                attention_mask=tensor_attention_mask
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            
        # 5. Ghép nhãn BIO từ predictions về cấp độ word
        # Bắt đầu duyệt từ index 1 (bỏ qua <s>)
        current_idx = 1
        word_predictions = []
        for w_len in word_subword_lengths:
            if w_len == 0:
                word_predictions.append("O")
                continue
                
            # Lấy nhãn của token con đầu tiên
            if current_idx < len(predictions):
                pred_id = predictions[current_idx]
                pred_label = ID2LABEL.get(pred_id, "O")
                word_predictions.append(pred_label)
            else:
                word_predictions.append("O")
                
            current_idx += w_len
            
        # Gom cụm các word theo nhãn BIO thành các trường thông tin tương ứng
        raw_pred = {
            "store_name": "",
            "date": "",
            "total": "",
            "address": ""
        }
        for word, label in zip(words, word_predictions):
            if label != "O":
                field_name = label[2:].lower()
                if field_name in raw_pred:
                    if raw_pred[field_name]:
                        raw_pred[field_name] += " " + word
                    else:
                        raw_pred[field_name] = word
                        
        # 6. Chuẩn hoá kết quả
        norm_pred = postprocess_extracted_fields(raw_pred)
        
        end_time = time.time()
        latency_cached = (end_time - start_model) * 1000
        latency_e2e = (end_time - start_e2e) * 1000
        
        return {
            "method": "layoutxlm",
            "prediction": raw_pred,
            "normalized_prediction": norm_pred,
            "raw_output": None,
            "latency_cached_ms": latency_cached,
            "latency_e2e_ms": latency_e2e,
            "status": "ok",
            "error": None,
            "words": flat_words,
            "word_labels": word_predictions
        }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Chạy suy luận LayoutXLM trên tập test.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/layoutxlm/receipt_ie/ocr_cache/best_model",
        help="Đường dẫn đến checkpoint tốt nhất của LayoutXLM"
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
        default="outputs/predictions/layoutxlm_test.jsonl",
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
    
    extractor = LayoutXLMExtractor()
    extractor.load(args.checkpoint)
    
    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
                
    if args.limit is not None:
        samples = samples[:args.limit]
        
    print(f"Running LayoutXLM inference on {len(samples)} samples...")
    count = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc="LayoutXLM Inference"):
            sample_id = sample.get("id")
            image_path_str = sample.get("image_path")
            
            prediction_record = {
                "id": sample_id,
                "method": "layoutxlm",
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
            
    print(f"LayoutXLM inference completed. Saved {count} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
