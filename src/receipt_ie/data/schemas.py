from abc import ABC, abstractmethod
from typing import Dict, Any, List
from PIL import Image

# Các trường thông tin mục tiêu cần trích xuất
FIELDS = ["store_name", "date", "total", "address"]

# Tập hợp các nhãn BIO cho LayoutXLM
LABELS = [
    "O",
    "B-STORE_NAME", "I-STORE_NAME",
    "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS"
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

class BaseExtractor(ABC):
    """
    Interface chung cho các phương pháp trích xuất thông tin biên lai.
    Đảm bảo tính nhất quán khi đánh giá và chạy Web Demo.
    """
    
    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """
        Nạp mô hình hoặc các cấu hình luật từ checkpoint_path.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Thực hiện suy luận từ ảnh đầu vào.
        
        Trả về kết quả ở dạng:
        {
            "store_name": "Tên cửa hàng hoặc rỗng",
            "date": "YYYY-MM-DD hoặc rỗng",
            "total": "Chuỗi chỉ chứa chữ số hoặc rỗng",
            "address": "Địa chỉ hoặc rỗng"
        }
        """
        pass
