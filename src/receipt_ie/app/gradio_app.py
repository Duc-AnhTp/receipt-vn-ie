import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# Thêm src vào sys.path nếu cần để chạy trực tiếp script
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from receipt_ie.data.schemas import BaseExtractor
from receipt_ie.inference.pipeline import get_extractor
from receipt_ie.inference.postprocess_json import postprocess_extracted_fields
from receipt_ie.inference.mock_extractor import MockExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Bảng màu trực quan cho nhãn BIO
COLOR_MAP = {
    "STORE_NAME": (239, 68, 68),  # Đỏ (Red)
    "DATE": (34, 197, 94),        # Xanh lá (Green)
    "TOTAL": (249, 115, 22),       # Cam (Orange)
    "ADDRESS": (168, 85, 247),    # Tím (Purple)
}


def draw_ocr_boxes(image: Image.Image, words: list) -> Image.Image:
    """Vẽ bounding boxes của toàn bộ text phát hiện được bởi OCR."""
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    for w in words:
        bbox = w["bbox"] # [xmin, ymin, xmax, ymax]
        draw.rectangle(bbox, outline=(59, 130, 246), width=2) # Xanh dương
    return draw_img


def draw_bio_predictions(image: Image.Image, words: list, labels: list) -> Image.Image:
    """Vẽ bounding boxes có màu sắc tương ứng cho các thực thể được LayoutXLM gán nhãn BIO."""
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    
    for w, label in zip(words, labels):
        if label == "O":
            continue
        field = label[2:] # Lấy tên trường, ví dụ STORE_NAME
        color = COLOR_MAP.get(field, (107, 114, 128)) # Xám mặc định
        bbox = w["bbox"]
        draw.rectangle(bbox, outline=color, width=3)
    return draw_img


def load_application_config() -> dict:
    config_path = project_root / "configs/app.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "models": {
            "donut_checkpoint": "checkpoints/donut/receipt_ie/final",
            "layoutxlm_checkpoint": "checkpoints/layoutxlm/receipt_ie/final",
            "baseline_rules": "configs/baseline.yaml"
        },
        "app": {
            "title": "Receipt VN IE",
            "description": "Trích xuất biên lai tiếng Việt",
            "fast_ocr_by_default": False
        }
    }


# Load cấu hình ứng dụng
app_config = load_application_config()

# Tránh nạp model tại top-level khi import (Lazy-loading)
EXTRACTORS = {"baseline": None, "donut": None, "layoutxlm": None}
MOCK_FLAGS = {"donut": False, "layoutxlm": False}

def get_baseline_extractor() -> BaseExtractor:
    if EXTRACTORS["baseline"] is None:
        logger.info("Initializing baseline extractor (lazy)...")
        baseline = get_extractor("baseline", project_root=str(project_root))
        EXTRACTORS["baseline"] = baseline
    return EXTRACTORS["baseline"]

def get_donut_extractor() -> BaseExtractor:
    if EXTRACTORS["donut"] is None:
        logger.info("Initializing donut extractor (lazy)...")
        donut_path = project_root / app_config["models"]["donut_checkpoint"]
        try:
            if not donut_path.exists() or not os.listdir(donut_path):
                raise FileNotFoundError("Donut checkpoint folder empty or not exists.")
            donut = get_extractor("donut", checkpoint_path=str(donut_path), project_root=str(project_root))
            MOCK_FLAGS["donut"] = False
        except Exception as e:
            logger.warning(f"Không thể nạp Donut checkpoint thực tế ({e}). Sử dụng Mock Donut.")
            baseline = get_baseline_extractor()
            donut = MockExtractor("donut", baseline)
            MOCK_FLAGS["donut"] = True
        EXTRACTORS["donut"] = donut
    return EXTRACTORS["donut"]

def get_layoutxlm_extractor() -> BaseExtractor:
    if EXTRACTORS["layoutxlm"] is None:
        logger.info("Initializing layoutxlm extractor (lazy)...")
        layoutxlm_path = project_root / app_config["models"]["layoutxlm_checkpoint"]
        try:
            if not layoutxlm_path.exists() or not os.listdir(layoutxlm_path):
                raise FileNotFoundError("LayoutXLM checkpoint folder empty or not exists.")
            layoutxlm = get_extractor("layoutxlm", checkpoint_path=str(layoutxlm_path), project_root=str(project_root))
            MOCK_FLAGS["layoutxlm"] = False
        except Exception as e:
            logger.warning(f"Không thể nạp LayoutXLM checkpoint thực tế ({e}). Sử dụng Mock LayoutXLM.")
            baseline = get_baseline_extractor()
            layoutxlm = MockExtractor("layoutxlm", baseline)
            MOCK_FLAGS["layoutxlm"] = True
        EXTRACTORS["layoutxlm"] = layoutxlm
    return EXTRACTORS["layoutxlm"]

# Kiểm tra mockup ban đầu dựa trên sự tồn tại của checkpoint (không load model)
def check_checkpoint_exists(checkpoint_key: str) -> bool:
    checkpoint_path = project_root / app_config["models"][checkpoint_key]
    return checkpoint_path.exists() and len(os.listdir(checkpoint_path)) > 0

is_donut_mock_initial = not check_checkpoint_exists("donut_checkpoint")
is_layoutxlm_mock_initial = not check_checkpoint_exists("layoutxlm_checkpoint")


def handle_demo_run(image: Image.Image, model_name: str, ocr_mode: str) -> Tuple[Dict[str, Any], str]:
    """Xử lý suy luận cho tab Demo."""
    if image is None:
        return {}, "Vui lòng tải ảnh lên."
        
    # Lazy load extractor tương ứng trước
    if model_name == "Baseline (Rule-based)":
        extractor = get_baseline_extractor()
    elif model_name == "Donut (OCR-free)":
        extractor = get_donut_extractor()
    else: # VietOCR + LayoutXLM
        extractor = get_layoutxlm_extractor()

    # Cấu hình OCR Mode động cho các extractor dùng OCR đã nạp
    rec_model = "vgg_seq2seq" if ocr_mode == "Fast (vgg_seq2seq)" else "vgg_transformer"
    for name in ["baseline", "layoutxlm"]:
        ext = EXTRACTORS.get(name)
        if ext is not None and hasattr(ext, "recognizer") and ext.recognizer is not None:
            # Nếu config hiện tại khác với yêu cầu, ta chuyển đổi model của VietOCR
            from receipt_ie.ocr.recognize_vietocr import load_vietocr_model
            current_config = getattr(ext.recognizer, "config_name", None)
            if current_config != rec_model:
                logger.info(f"Switching VietOCR model config to: {rec_model}")
                gpu_avail = torch.cuda.is_available()
                ext.recognizer = load_vietocr_model(config_name=rec_model, use_gpu=gpu_avail)
                ext.recognizer.config_name = rec_model
                
    start_time = time.time()
    res = extractor.predict(image)
    e2e_time = (time.time() - start_time) * 1000
    
    # Lấy thông tin hiển thị latency
    lat_cached = res.get("latency_cached_ms", 0.0)
    lat_e2e = res.get("latency_e2e_ms", e2e_time)
    
    latency_info = (
        f"**Thời gian suy luận (End-to-End):** {lat_e2e:.2f} ms\n"
        f"**Thời gian xử lý của Mô hình (Model-only):** {lat_cached:.2f} ms\n"
    )
    if model_name == "Donut (OCR-free)":
        latency_info = f"**Thời gian suy luận (Donut E2E):** {res.get('latency_ms', e2e_time):.2f} ms"
        
    # Hiển thị ghi chú nếu đang dùng Mock Model
    is_mock = res.get("is_mock", False)
    if is_mock:
        latency_info += "\n*(Lưu ý: Đang chạy ở chế độ Mock Model do thiếu checkpoint)*"
        
    return res.get("normalized_prediction", {}), latency_info


def handle_compare_run(image: Image.Image, ocr_mode: str) -> Tuple[Image.Image, Image.Image, str, str, str, str]:
    """Xử lý so sánh side-by-side và trích xuất debug cho tab Debug & Compare."""
    if image is None:
        empty_img = Image.new("RGB", (200, 200), color="white")
        return empty_img, empty_img, "", "", "", "Vui lòng tải ảnh lên."
        
    # Lazy load tất cả extractors phục vụ debug & so sánh
    baseline_extractor = get_baseline_extractor()
    donut_extractor = get_donut_extractor()
    layoutxlm_extractor = get_layoutxlm_extractor()

    # Cấu hình OCR Mode
    rec_model = "vgg_seq2seq" if ocr_mode == "Fast (vgg_seq2seq)" else "vgg_transformer"
    for name in ["baseline", "layoutxlm"]:
        ext = EXTRACTORS.get(name)
        if ext is not None and hasattr(ext, "recognizer") and ext.recognizer is not None:
            from receipt_ie.ocr.recognize_vietocr import load_vietocr_model
            current_config = getattr(ext.recognizer, "config_name", None)
            if current_config != rec_model:
                gpu_avail = torch.cuda.is_available()
                ext.recognizer = load_vietocr_model(config_name=rec_model, use_gpu=gpu_avail)
                ext.recognizer.config_name = rec_model
                
    # 1. Chạy 3 mô hình
    res_base = baseline_extractor.predict(image)
    res_donut = donut_extractor.predict(image)
    res_layout = layoutxlm_extractor.predict(image)
    
    # 2. Tạo ảnh visualize OCR
    words = res_base.get("words", [])
    ocr_visualized = draw_ocr_boxes(image, words)
    
    # 3. Tạo ảnh visualize BIO Labels (LayoutXLM)
    layout_words = res_layout.get("words", [])
    layout_labels = res_layout.get("word_labels", [])
    bio_visualized = draw_bio_predictions(image, layout_words, layout_labels)
    
    # 4. Trích xuất Donut raw sequence
    donut_raw = res_donut.get("raw_output", "N/A")
    
    # 5. Tạo bảng so sánh kết quả Markdown
    pred_base = res_base.get("normalized_prediction", {})
    pred_donut = res_donut.get("normalized_prediction", {})
    pred_layout = res_layout.get("normalized_prediction", {})
    
    compare_table = f"""
| Trường Thông Tin | Baseline (Rule-based) | Donut (OCR-free) | VietOCR + LayoutXLM |
| :--- | :--- | :--- | :--- |
| **Store Name** | {pred_base.get('store_name', '')} | {pred_donut.get('store_name', '')} | {pred_layout.get('store_name', '')} |
| **Date** | {pred_base.get('date', '')} | {pred_donut.get('date', '')} | {pred_layout.get('date', '')} |
| **Total** | {pred_base.get('total', '')} | {pred_donut.get('total', '')} | {pred_layout.get('total', '')} |
| **Address** | {pred_base.get('address', '')} | {pred_donut.get('address', '')} | {pred_layout.get('address', '')} |
"""
    
    # 6. Bảng so sánh Latency
    lat_b_e2e = res_base.get("latency_e2e_ms", 0.0)
    lat_d_e2e = res_donut.get("latency_ms", res_donut.get("latency_e2e_ms", 0.0))
    lat_l_e2e = res_layout.get("latency_e2e_ms", 0.0)
    
    lat_b_cached = res_base.get("latency_cached_ms", 0.0)
    lat_d_cached = res_donut.get("latency_ms", res_donut.get("latency_cached_ms", 0.0))
    lat_l_cached = res_layout.get("latency_cached_ms", 0.0)
    
    latency_table = f"""
| Mô hình | Latency Cached (Model-only) | Latency E2E (Gồm OCR) | Chế độ chạy |
| :--- | :--- | :--- | :--- |
| **Baseline** | {lat_b_cached:.1f} ms | {lat_b_e2e:.1f} ms | Thực tế |
| **Donut** | {lat_d_cached:.1f} ms | {lat_d_e2e:.1f} ms | {"Mock" if res_donut.get("is_mock") else "Thực tế"} |
| **LayoutXLM** | {lat_l_cached:.1f} ms | {lat_l_e2e:.1f} ms | {"Mock" if res_layout.get("is_mock") else "Thực tế"} |
"""

    status_info = []
    if res_donut.get("is_mock"):
        status_info.append("Donut (Mock Mode)")
    if res_layout.get("is_mock"):
        status_info.append("LayoutXLM (Mock Mode)")
        
    status_str = f"Trạng thái: OK. "
    if status_info:
        status_str += f"Đang chạy mockup cho: {', '.join(status_info)}."

    return ocr_visualized, bio_visualized, donut_raw, compare_table, latency_table, status_str


# Tạo giao diện Blocks Gradio
theme = gr.themes.Soft(primary_hue="teal", secondary_hue="indigo")

with gr.Blocks(theme=theme, title="Receipt VN Information Extraction") as demo:
    gr.HTML(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="background: linear-gradient(90deg, #14b8a6, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800;">
                Receipt VN IE
            </h1>
            <p style="font-size: 1.1rem; color: #4b5563;">
                Hệ thống trích xuất thông tin biên lai tiếng Việt: Donut vs VietOCR + LayoutXLM
            </p>
        </div>
        """
    )
    
    # Hiển thị cảnh báo mockup nếu có
    if is_donut_mock_initial or is_layoutxlm_mock_initial:
        mocks = []
        if is_donut_mock_initial: mocks.append("Donut")
        if is_layoutxlm_mock_initial: mocks.append("LayoutXLM")
        gr.Markdown(
            f"⚠️ **Lưu ý:** Không tìm thấy checkpoints huấn luyện cho **{', '.join(mocks)}**. "
            f"Hệ thống tự động kích hoạt **Chế độ Mock Model** (sử dụng baseline OCR làm backend mô phỏng) để đảm bảo giao diện hoạt động bình thường."
        )
        
    with gr.Row():
        with gr.Column(scale=1):
            # Input image
            input_image = gr.Image(type="pil", label="Tải ảnh biên lai lên")
            
            # Cấu hình OCR Mode
            ocr_mode = gr.Dropdown(
                choices=["Accurate (vgg_transformer)", "Fast (vgg_seq2seq)"],
                value="Accurate (vgg_transformer)",
                label="Chế độ OCR (VietOCR)",
                info="Fast Mode tối ưu hóa tốc độ nhưng độ chính xác có thể giảm so với Accurate Mode."
            )
            
        with gr.Column(scale=2):
            with gr.Tab("Tab 1: Demo Trích Xuất"):
                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=["Baseline (Rule-based)", "Donut (OCR-free)", "VietOCR + LayoutXLM"],
                        value="VietOCR + LayoutXLM",
                        label="Chọn mô hình trích xuất"
                    )
                
                run_btn = gr.Button("Bắt đầu trích xuất", variant="primary")
                
                with gr.Row():
                    output_json = gr.JSON(label="Thông tin trích xuất (JSON)")
                
                with gr.Row():
                    latency_display = gr.Markdown("### Thông tin hiệu năng\nNhấn nút để chạy suy luận.")
                    
                run_btn.click(
                    fn=handle_demo_run,
                    inputs=[input_image, model_selector, ocr_mode],
                    outputs=[output_json, latency_display]
                )
                
            with gr.Tab("Tab 2: Debug & So Sánh"):
                compare_btn = gr.Button("Chạy So Sánh & Debug", variant="secondary")
                
                with gr.Row():
                    with gr.Column():
                        ocr_image_display = gr.Image(label="Vùng chữ phát hiện bởi OCR (Baseline/LayoutXLM)")
                    with gr.Column():
                        bio_image_display = gr.Image(label="Thực thể BIO được phân loại (LayoutXLM)")
                
                with gr.Row():
                    gr.Markdown("### So Sánh Kết Quả Trích Xuất")
                with gr.Row():
                    compare_output_table = gr.Markdown("Nhấn nút so sánh để tạo bảng.")
                    
                with gr.Row():
                    gr.Markdown("### So Sánh Latency (ms)")
                with gr.Row():
                    latency_output_table = gr.Markdown("Nhấn nút so sánh để tạo bảng hiệu năng.")
                    
                with gr.Row():
                    donut_raw_display = gr.Textbox(label="Chuỗi Tag-Sequence thô của Donut", lines=3, max_lines=5)
                    
                status_display = gr.Markdown("*Chưa chạy so sánh.*")
                
                compare_btn.click(
                    fn=handle_compare_run,
                    inputs=[input_image, ocr_mode],
                    outputs=[
                        ocr_image_display,
                        bio_image_display,
                        donut_raw_display,
                        compare_output_table,
                        latency_output_table,
                        status_display
                    ]
                )

if __name__ == "__main__":
    # Đọc host và port từ cấu hình
    server_cfg = app_config.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 7860)
    share = server_cfg.get("share", False)
    
    logger.info(f"Khởi chạy Gradio Server tại http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, share=share)
