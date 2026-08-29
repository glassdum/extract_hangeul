"""명암 보정: CLAHE, Gamma, Contrast Stretch (문서 "이미지 전처리" 표)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def apply_clahe(
    image: Image.Image, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> Image.Image:
    gray = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(gray)
    return Image.fromarray(cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB))


def stretch_contrast(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L")).astype(np.float32)
    low, high = np.percentile(gray, (2, 98))
    if high <= low:
        return image.convert("RGB")
    stretched = np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB))


def adjust_gamma(image: Image.Image, gamma: float = 1.5) -> Image.Image:
    array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    corrected = np.power(array, 1.0 / gamma)
    return Image.fromarray((corrected * 255.0).astype(np.uint8))
