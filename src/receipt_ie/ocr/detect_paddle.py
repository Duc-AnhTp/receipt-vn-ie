"""
Wrapper cho PaddleOCR text detection.
Chỉ dùng phần detection (phát hiện vùng chữ), không dùng recognition của Paddle.
"""
import numpy as np
from pathlib import Path
from PIL import Image


def load_paddle_detector(use_gpu: bool = True, use_angle_cls: bool = True, lang: str = "vi"):
    """
    Khởi tạo PaddleOCR detector.
    Chỉ bật det=True, rec=False để lấy bounding boxes.
    """
    from paddleocr import PaddleOCR

    detector = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=lang,
        use_gpu=use_gpu,
        det=True,
        rec=False,
        show_log=False,
    )
    return detector


def detect_text_regions(detector, image_path: str) -> list[dict]:
    """
    Phát hiện các vùng chứa chữ trong ảnh.

    Returns:
        List[dict]: Mỗi phần tử chứa:
            - "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - "bbox": [x0, y0, x1, y1] (bounding rect)
            - "confidence": float (det confidence)
    """
    result = detector.ocr(image_path, det=True, rec=False, cls=False)

    regions = []
    if not result or not result[0]:
        return regions

    for box_info in result[0]:
        # PaddleOCR trả về polygon dạng [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        polygon = box_info
        if isinstance(box_info, (list, np.ndarray)):
            polygon = np.array(polygon).tolist()

        # Tính bounding rect
        xs = [pt[0] for pt in polygon]
        ys = [pt[1] for pt in polygon]
        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        regions.append({
            "polygon": [[int(p[0]), int(p[1])] for p in polygon],
            "bbox": bbox,
        })

    return regions


def crop_region(image: Image.Image, bbox: list[int], padding: int = 2) -> Image.Image:
    """
    Cắt vùng ảnh theo bounding box [x0, y0, x1, y1] với padding.
    """
    w, h = image.size
    x0 = max(0, bbox[0] - padding)
    y0 = max(0, bbox[1] - padding)
    x1 = min(w, bbox[2] + padding)
    y1 = min(h, bbox[3] + padding)
    return image.crop((x0, y0, x1, y1))
