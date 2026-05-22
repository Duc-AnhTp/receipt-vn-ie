from receipt_ie.data.schemas import BaseExtractor

def get_extractor(
    method: str,
    checkpoint_path: str = "",
    ocr_config_path: str = "configs/ocr.yaml",
    project_root: str = "."
) -> BaseExtractor:
    """
    Factory function khởi tạo và nạp extractor tương ứng theo tên phương pháp.
    Các phương pháp hỗ trợ: 'baseline', 'donut', 'layoutxlm'
    """
    method_lower = method.lower()
    
    if method_lower == "baseline":
        from receipt_ie.inference.infer_baseline import BaselineExtractor
        extractor = BaselineExtractor(ocr_config_path=ocr_config_path, project_root=project_root)
        extractor.load(checkpoint_path)
        return extractor
        
    elif method_lower == "donut":
        from receipt_ie.inference.infer_donut import DonutExtractor
        extractor = DonutExtractor()
        extractor.load(checkpoint_path)
        return extractor
        
    elif method_lower == "layoutxlm":
        from receipt_ie.inference.infer_layoutxlm import LayoutXLMExtractor
        extractor = LayoutXLMExtractor(ocr_config_path=ocr_config_path, project_root=project_root)
        extractor.load(checkpoint_path)
        return extractor
        
    else:
        raise ValueError(f"Không hỗ trợ phương thức trích xuất: '{method}'. Phải là baseline, donut, hoặc layoutxlm.")
