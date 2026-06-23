import torch
# Đảm bảo import torch đầu tiên trên Windows để tránh DLL collision
import time
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
from transformers import AutoTokenizer, LayoutLMv2ForTokenClassification

from receipt_ie.data.schemas import BaseExtractor, ID2LABEL
from receipt_ie.ocr.detect_paddle import load_paddle_detector, detect_text_regions, crop_region
from receipt_ie.ocr.recognize_vietocr import load_vietocr_model, recognize_regions
from receipt_ie.ocr.reading_order import sort_reading_order
from receipt_ie.data.build_layoutxlm_labels import normalize_bbox
from receipt_ie.inference.postprocess_json import postprocess_extracted_fields
from receipt_ie.inference.artifact_metadata import write_inference_sidecar
from receipt_ie.inference.cache_coordinates import resolve_cached_image_path
from receipt_ie.postprocess.total_extractor import is_probably_phone, is_probably_tax_code, is_too_small_for_total
from receipt_ie.data.normalize_text import normalize_money, normalize_date

EMPTY_FIELDS = {"store_name": "", "date": "", "total": "", "address": ""}


def _empty_fields() -> Dict[str, str]:
    return EMPTY_FIELDS.copy()


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

    def load(self, checkpoint_path: str, init_ocr: bool = True) -> None:
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
        if init_ocr and (self.detector is None or self.recognizer is None):
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
            rec_gpu = torch.cuda.is_available()
            self.recognizer = load_vietocr_model(config_name=rec_config, use_gpu=rec_gpu)
        print("OCR engines initialized.")

    def predict_from_ocr(self, image: Image.Image, words: List[str], boxes: List[List[int]]) -> Dict[str, Any]:
        """
        Thực hiện dự đoán trực tiếp sử dụng token và bboxes đã trích xuất từ OCR cache.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Mô hình chưa được nạp. Vui lòng gọi hàm load() trước.")
            
        start_model = time.time()
        width, height = image.size
        
        # 1. Chuẩn bị token và bbox cho LayoutXLM
        input_ids = []
        bbox = []
        
        # Token bắt đầu: <s>
        input_ids.append(self.tokenizer.bos_token_id)
        bbox.append([0, 0, 0, 0])
        
        # Phân rã words thành subwords và theo vết token con đầu tiên
        word_subword_lengths = []
        for word, box in zip(words, boxes):
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
        
        # 2. Chạy mô hình LayoutXLM dự đoán
        with torch.no_grad():
            outputs = self.model(
                input_ids=tensor_input_ids,
                bbox=tensor_bbox,
                image=tensor_image,
                attention_mask=tensor_attention_mask
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            
        # 3. Ghép nhãn BIO từ predictions về cấp độ word
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
            
        # 4. Gom cụm các word theo nhãn BIO thành các thực thể (entity spans)
        spans = []
        current_span = None
        for i, (word, box, label) in enumerate(zip(words, boxes, word_predictions)):
            if label.startswith("B-"):
                if current_span:
                    spans.append(current_span)
                current_span = {"field": label[2:].lower(), "words": [word], "boxes": [box]}
            elif label.startswith("I-"):
                field = label[2:].lower()
                if current_span and current_span["field"] == field:
                    current_span["words"].append(word)
                    current_span["boxes"].append(box)
                else:
                    if current_span:
                        spans.append(current_span)
                    current_span = {"field": field, "words": [word], "boxes": [box]}
            else: # O
                if current_span:
                    spans.append(current_span)
                current_span = None
        if current_span:
            spans.append(current_span)
            
        # Tạo chuỗi text và bounding box cho các spans
        processed_spans = []
        for span in spans:
            span_text = " ".join(span["words"]).strip()
            # Tính bounding box bao quanh span
            xs = [b[0] for b in span["boxes"]] + [b[2] for b in span["boxes"]]
            ys = [b[1] for b in span["boxes"]] + [b[3] for b in span["boxes"]]
            span_box = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]
            processed_spans.append({
                "field": span["field"],
                "text": span_text,
                "bbox": span_box,
                "height": max(1, span_box[3] - span_box[1])
            })
            
        # Gom nhóm cụ thể từng trường theo heuristics
        # 1. STORE_NAME: Lấy span đầu tiên
        store_spans = [s for s in processed_spans if s["field"] == "store_name"]
        store_val = store_spans[0]["text"] if store_spans else ""
        
        # 2. DATE: Lấy span đầu tiên hợp lệ sau khi thử normalize
        date_spans = [s for s in processed_spans if s["field"] == "date"]
        date_val = ""
        for s in date_spans:
            norm_d = normalize_date(s["text"])
            if norm_d:
                date_val = s["text"]
                break
        if not date_val and date_spans:
            date_val = date_spans[0]["text"]
            
        # 3. TOTAL: Heuristics lọc phone/MST/mã hóa đơn và chọn lớn nhất
        total_spans = [s for s in processed_spans if s["field"] == "total"]
        valid_totals = []
        for s in total_spans:
            cleaned = normalize_money(s["text"])
            if cleaned and cleaned.isdigit():
                val = int(cleaned)
                if not is_probably_phone(cleaned) and not is_probably_tax_code(cleaned) and not is_too_small_for_total(val):
                    valid_totals.append((s, val))
        
        total_val = ""
        if valid_totals:
            total_val = str(max(valid_totals, key=lambda x: x[1])[1])
        elif total_spans:
            total_val = total_spans[0]["text"]
            
        # 4. ADDRESS: Gom nhóm nối y-distance (<= 1.5 lần chiều cao dòng)
        address_spans = [s for s in processed_spans if s["field"] == "address"]
        address_spans = sorted(address_spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
        
        address_blocks = []
        current_block = []
        for s in address_spans:
            if not current_block:
                current_block.append(s)
            else:
                prev = current_block[-1]
                y_dist = s["bbox"][1] - prev["bbox"][3]
                avg_h = (prev["height"] + s["height"]) / 2
                
                # Nếu khoảng cách y nhỏ, gom chung block
                if y_dist <= 1.5 * avg_h:
                    current_block.append(s)
                else:
                    address_blocks.append(current_block)
                    current_block = [s]
        if current_block:
            address_blocks.append(current_block)
            
        # Chọn block ADDRESS tối ưu nhất (chứa keyword địa chỉ hoặc block dài nhất)
        best_address_text = ""
        if address_blocks:
            block_candidates = []
            for block in address_blocks:
                block_text = " ".join([s["text"] for s in block]).strip()
                has_kw = any(kw in block_text.lower() for kw in ["địa chỉ", "dia chi", "address", "đ/c", "dc"])
                score = len(block_text) + (500 if has_kw else 0)
                block_candidates.append((score, block_text))
            best_address_text = max(block_candidates, key=lambda x: x[0])[1]
            
        raw_pred = {
            "store_name": store_val,
            "date": date_val,
            "total": total_val,
            "address": best_address_text
        }
        
        # 5. Chuẩn hoá kết quả
        norm_pred = postprocess_extracted_fields(raw_pred)
        
        end_time = time.time()
        model_ms = (end_time - start_model) * 1000
        
        # Tạo dữ liệu debug BIO
        entities_debug = []
        for span in processed_spans:
            entities_debug.append({
                "field": span["field"],
                "text": span["text"],
                "bbox": span["bbox"]
            })
            
        return {
            "latency_ocr_ms": 0.0,
            "latency_model_ms": model_ms,
            "latency_postprocess_ms": 0.0,  # postprocess lồng trong model_ms
            "latency_e2e_ms": model_ms,
            "prediction": raw_pred,
            "normalized_prediction": norm_pred,
            "status": "ok",
            "error": None,
            "layoutxlm_debug": {
                "entities": entities_debug
            }
        }

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Thực hiện suy luận từ PIL Image. Sử dụng runtime cache OCR.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Mô hình chưa được nạp. Vui lòng gọi hàm load() trước.")
            
        if self.detector is None or self.recognizer is None:
            self._init_ocr()
            
        from receipt_ie.ocr.cache_manager import get_or_build_ocr
        
        start_ocr = time.time()
        ocr_data = get_or_build_ocr(
            image=image,
            detector=self.detector,
            recognizer=self.recognizer,
            preprocess_profile="resize",
            ocr_config_path=str(self.ocr_config_path),
            cache_dir=str(self.project_root / "outputs/runtime_ocr_cache")
        )
        ocr_duration = (time.time() - start_ocr) * 1000
        
        latency_ocr_ms = ocr_duration if ocr_duration > 15.0 else 0.0
        
        flat_words = ocr_data.get("words", [])
        words = [w["text"] for w in flat_words]
        boxes = [w["bbox"] for w in flat_words]
        
        cached_image_path = resolve_cached_image_path(
            ocr_data,
            project_root=self.project_root,
        )
        model_image = (
            Image.open(cached_image_path).convert("RGB")
            if cached_image_path
            else image
        )
        res = self.predict_from_ocr(model_image, words, boxes)
        res["latency_ocr_ms"] = latency_ocr_ms
        res["latency_e2e_ms"] = latency_ocr_ms + res["latency_model_ms"]
        return res


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
        default="outputs/predictions/layoutxlm_test_v2.jsonl",
        help="Đường dẫn file JSONL đầu ra"
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow overwriting an existing output artifact.",
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
    from tqdm import tqdm
    
    args = parse_args()
    
    test_file = Path(args.test_jsonl)
    if not test_file.exists():
        print(f"Không tìm thấy file test: {test_file}")
        return
        
    out_file = Path(args.output_jsonl)
    if out_file.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {out_file}. "
            "Choose a new path or pass --allow_overwrite explicitly."
        )
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
            ocr_cache_path_str = sample.get("ocr_cache_path")
            
            prediction_record = {
                "id": sample_id,
                "method": "layoutxlm",
                "prediction": {},
                "normalized_prediction": {},
                "raw_output": None,
                "latency_ocr_ms": 0.0,
                "latency_model_ms": 0.0,
                "latency_postprocess_ms": 0.0,
                "latency_e2e_ms": 0.0,
                "status": "ok",
                "error": None
            }
            
            try:
                if not image_path_str or not os.path.exists(image_path_str):
                    raise FileNotFoundError(f"Image not found at {image_path_str}")
                    
                img = Image.open(image_path_str).convert("RGB")
                
                # Thử đọc OCR cache offline trước để tăng tốc evaluation
                ocr_data = None
                ocr_start_time = 0
                ocr_end_time = 0
                
                if ocr_cache_path_str and os.path.exists(ocr_cache_path_str):
                    with open(ocr_cache_path_str, "r", encoding="utf-8") as cache_f:
                        ocr_data = json.load(cache_f)
                else:
                    ocr_start_time = time.time()
                    det_eng, rec_eng = extractor.detector, extractor.recognizer
                    regions = detect_text_regions(det_eng, image_path_str)
                    
                    if regions:
                        cropped_imgs = [crop_region(img, r["bbox"], padding=2) for r in regions]
                        texts = recognize_regions(rec_eng, cropped_imgs, batch_size=16)
                        for r, text in zip(regions, texts):
                            r["text"] = text.strip()
                        regions = [r for r in regions if r["text"]]
                        
                        flat_words, grouped_lines = sort_reading_order(regions, y_threshold=extractor.y_threshold)
                        ocr_data = {
                            "words": flat_words
                        }
                    else:
                        ocr_data = {"words": []}
                    ocr_end_time = time.time()
                    
                ocr_latency_ms = (ocr_end_time - ocr_start_time) * 1000.0 if ocr_start_time > 0 else 0.0
                
                flat_words = ocr_data.get("words", [])
                words = [w["text"] for w in flat_words]
                boxes = [w["bbox"] for w in flat_words]
                
                cached_image_path = resolve_cached_image_path(
                    ocr_data,
                    cache_path=ocr_cache_path_str,
                    project_root=extractor.project_root,
                )
                model_image = (
                    Image.open(cached_image_path).convert("RGB")
                    if cached_image_path
                    else img
                )
                res = extractor.predict_from_ocr(model_image, words, boxes)
                
                prediction_record["prediction"] = res["prediction"]
                prediction_record["normalized_prediction"] = res["normalized_prediction"]
                prediction_record["latency_ocr_ms"] = round(ocr_latency_ms, 2)
                prediction_record["latency_model_ms"] = round(res["latency_model_ms"], 2)
                prediction_record["latency_postprocess_ms"] = round(res["latency_postprocess_ms"], 2)
                prediction_record["latency_e2e_ms"] = round(ocr_latency_ms + res["latency_model_ms"], 2)
                
                if "layoutxlm_debug" in res:
                    prediction_record["layoutxlm_debug"] = res["layoutxlm_debug"]
                
            except Exception as e:
                print(f"Error evaluating sample {sample_id}: {e}")
                prediction_record["status"] = "error"
                prediction_record["error"] = str(e)
                
            out_f.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"LayoutXLM inference completed. Saved {count} predictions to {args.output_jsonl}")
    sidecar = write_inference_sidecar(
        out_file,
        method="layoutxlm",
        checkpoint=args.checkpoint,
        device=extractor.device,
        prediction_count=count,
        inference_arguments={
            "max_length": extractor.max_length,
            "image_source": "ocr_cache_preprocessed_image_when_available",
        },
    )
    print(f"Inference metadata saved to {sidecar}")


if __name__ == "__main__":
    main()
