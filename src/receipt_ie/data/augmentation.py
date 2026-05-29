import albumentations as A
import numpy as np
from PIL import Image

def get_donut_transforms():
    """
    Tăng cường dữ liệu cho Donut (bao gồm cả biến đổi hình học và biến đổi pixel).
    """
    return A.Compose([
        # Xoay nhẹ ảnh (giả lập hóa đơn chụp bị lệch)
        A.ShiftScaleRotate(
            shift_limit=0.03, 
            scale_limit=0.05, 
            rotate_limit=10, 
            border_mode=0, 
            value=(255, 255, 255), 
            p=0.5
        ),
        # Méo phối cảnh (giả lập góc chụp camera nghiêng)
        A.Perspective(scale=(0.02, 0.04), pad_val=(255, 255, 255), p=0.3),
        # Chỉnh độ sáng & độ tương phản (giả lập ánh sáng không đều)
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        # Làm mờ nhẹ (giả lập camera out-focus hoặc rung tay)
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # Tạo bóng đổ ngẫu nhiên đè lên ảnh
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_limit=1,
            shadow_dimension=5,
            p=0.2
        ),
    ])

def get_layoutxlm_transforms():
    """
    Tăng cường dữ liệu cho LayoutXLM (chỉ biến đổi pixel để giữ nguyên tọa độ bounding box).
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_limit=1,
            shadow_dimension=5,
            p=0.2
        ),
    ])

def apply_transforms(image: Image.Image, transform_pipeline) -> Image.Image:
    """Áp dụng pipeline biến đổi và trả về ảnh PIL."""
    image_np = np.array(image)
    augmented = transform_pipeline(image=image_np)
    return Image.fromarray(augmented["image"])
