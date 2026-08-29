"""불확실한 Crop 재판독을 위한 여백 포함 잘라내기."""

from __future__ import annotations

from PIL import Image

from common.types import BBox


def crop_with_padding(image: Image.Image, bbox: BBox, pad_ratio: float = 0.15) -> Image.Image:
    width, height = image.size
    pad_x = bbox.width * pad_ratio
    pad_y = bbox.height * pad_ratio

    x0 = max(0, int(bbox.x0 - pad_x))
    y0 = max(0, int(bbox.y0 - pad_y))
    x1 = min(width, int(bbox.x1 + pad_x) + 1)
    y1 = min(height, int(bbox.y1 + pad_y) + 1)

    if x1 <= x0 or y1 <= y0:
        return image.convert("RGB")
    return image.crop((x0, y0, x1, y1))
