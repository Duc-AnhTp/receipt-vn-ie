from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageOps


@dataclass
class PreprocessResult:
    image: Image.Image
    profile: str
    scale_x: float
    scale_y: float
    metadata: Dict[str, Any]


def load_receipt_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_long_side(image: Image.Image, max_long_side: int = 1600) -> tuple[Image.Image, float, float]:
    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image, 1.0, 1.0

    scale = max_long_side / long_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.BICUBIC)
    return resized, scale, scale


def preprocess_receipt_image(
    image: Image.Image,
    profile: str = "none",
    max_long_side: int = 1600,
) -> PreprocessResult:
    original_width, original_height = image.size
    metadata: Dict[str, Any] = {
        "original_width": original_width,
        "original_height": original_height,
        "max_long_side": max_long_side,
        "steps": [],
        "coordinate_transform": "identity",
    }

    if profile == "none":
        return PreprocessResult(image=image, profile=profile, scale_x=1.0, scale_y=1.0, metadata=metadata)

    if profile in {"resize", "ocr_best"}:
        processed, scale_x, scale_y = resize_long_side(image, max_long_side=max_long_side)
        if scale_x != 1.0 or scale_y != 1.0:
            metadata["steps"].append("resize_long_side")
            metadata["coordinate_transform"] = "scale"
        metadata["resized_width"] = processed.size[0]
        metadata["resized_height"] = processed.size[1]
        return PreprocessResult(processed, profile, scale_x, scale_y, metadata)

    if profile == "rectify":
        from receipt_ie.ocr.preprocess import rectify_document

        processed = rectify_document(image)
        metadata["steps"].append("rectify_document")
        metadata["rectified"] = True
        metadata["coordinate_transform"] = "unknown"
        processed, scale_x, scale_y = resize_long_side(processed, max_long_side=max_long_side)
        if scale_x != 1.0 or scale_y != 1.0:
            metadata["steps"].append("resize_long_side")
        metadata["resized_width"] = processed.size[0]
        metadata["resized_height"] = processed.size[1]
        return PreprocessResult(processed, profile, scale_x, scale_y, metadata)

    if profile == "binarize":
        from receipt_ie.ocr.preprocess import binarize_image

        processed, scale_x, scale_y = resize_long_side(image, max_long_side=max_long_side)
        if scale_x != 1.0 or scale_y != 1.0:
            metadata["steps"].append("resize_long_side")
            metadata["coordinate_transform"] = "scale"
        processed = binarize_image(processed)
        metadata["steps"].append("binarize_image")
        metadata["binarized"] = True
        metadata["resized_width"] = processed.size[0]
        metadata["resized_height"] = processed.size[1]
        return PreprocessResult(processed, profile, scale_x, scale_y, metadata)

    raise ValueError(f"Unknown preprocess profile: {profile}")
