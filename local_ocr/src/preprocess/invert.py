"""반전: Black-on-white / White-on-black (문서 "이미지 전처리" 표, 역상 텍스트용)."""

from __future__ import annotations

from PIL import Image, ImageOps


def invert(image: Image.Image) -> Image.Image:
    return ImageOps.invert(image.convert("RGB"))
