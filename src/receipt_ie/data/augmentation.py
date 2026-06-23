import albumentations as A
import numpy as np
from PIL import Image


def get_donut_transforms():
    """Donut augmentation for Vietnamese receipts with small accented text, tilted angles, and shadows."""
    return A.Compose([
        A.Affine(
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            scale=(0.97, 1.03),
            rotate=(-8, 8),
            border_mode=0,
            fill=255,
            p=0.4,
        ),
        A.Perspective(scale=(0.01, 0.025), p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
        A.GaussianBlur(blur_limit=(3, 3), p=0.08),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.15),
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            p=0.2,
        ),
    ])


def get_layoutxlm_transforms():
    """Pixel-only augmentation for LayoutXLM so OCR/annotation boxes remain valid."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            p=0.2,
        ),
    ])


def apply_transforms(image: Image.Image, transform_pipeline) -> Image.Image:
    image_np = np.array(image)
    augmented = transform_pipeline(image=image_np)
    return Image.fromarray(augmented["image"])
