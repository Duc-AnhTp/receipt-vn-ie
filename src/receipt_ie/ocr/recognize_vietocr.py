"""
Wrapper cho VietOCR text recognition.
Sử dụng các mô hình Transformer-based của VietOCR để nhận dạng chữ tiếng Việt từ các ảnh đã crop.
"""
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def load_vietocr_model(config_name: str = "vgg_transformer", use_gpu: bool = True):
    """
    Khởi tạo mô hình VietOCR Predictor.
    
    Args:
        config_name (str): Tên cấu hình VietOCR ("vgg_transformer" hoặc "vgg_seq2seq").
        use_gpu (bool): Có sử dụng GPU hay không.
    """
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

    logger.info(f"Loading VietOCR model '{config_name}' (GPU: {use_gpu})...")
    config = Cfg.load_config_from_name(config_name)
    config['device'] = 'cuda:0' if use_gpu else 'cpu'
    
    # Khởi tạo Predictor
    predictor = Predictor(config)
    return predictor


def recognize_regions(predictor, cropped_images: list[Image.Image], batch_size: int = 16) -> list[str]:
    """
    Nhận dạng chữ cho một danh sách các ảnh đã crop.
    Sử dụng predict_batch để tối ưu hiệu năng, fallback về predict từng ảnh nếu gặp lỗi.
    
    Args:
        predictor: Mô hình VietOCR Predictor đã load.
        cropped_images (list[PIL.Image.Image]): Danh sách ảnh vùng chữ.
        batch_size (int): Kích thước batch khi suy luận.
        
    Returns:
        list[str]: Danh sách chuỗi văn bản tương ứng.
    """
    if not cropped_images:
        return []

    texts = []
    
    # Thử sử dụng predict_batch nếu có sẵn
    if hasattr(predictor, 'predict_batch'):
        try:
            # Chia batch thủ công hoặc để predict_batch tự xử lý (vietocr nhận list và tự batch)
            # Thông thường, Predictor.predict_batch nhận list các PIL Image
            # Ta chia nhỏ theo batch_size đề phòng lỗi out of memory hoặc quá tải
            for i in range(0, len(cropped_images), batch_size):
                batch = cropped_images[i:i + batch_size]
                batch_texts = predictor.predict_batch(batch)
                texts.extend(batch_texts)
            return texts
        except Exception as e:
            logger.warning(f"Failed to use predict_batch, fallback to loop: {e}")
            texts = []

    # Fallback loop
    for img in cropped_images:
        try:
            text = predictor.predict(img)
            texts.append(text)
        except Exception as e:
            logger.error(f"Error predicting image region: {e}")
            texts.append("")
            
    return texts
