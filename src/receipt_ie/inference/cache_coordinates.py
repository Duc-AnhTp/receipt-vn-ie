"""Helpers for keeping OCR boxes and document images in one coordinate space."""

from pathlib import Path
from typing import Any, Dict


def resolve_cached_image_path(
    cache_data: Dict[str, Any],
    *,
    cache_path: str | Path | None = None,
    project_root: str | Path = ".",
) -> Path | None:
    """Resolve the image whose coordinate system matches cached OCR boxes."""
    value = cache_data.get("preprocessed_image_path")
    if not value:
        return None
    candidate = Path(value)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(Path(project_root) / candidate)
        if cache_path:
            candidates.append(Path(cache_path).parent / candidate.name)
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None
