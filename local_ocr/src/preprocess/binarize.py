"""이진화: Otsu, Adaptive Threshold (문서 "이미지 전처리" 표)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def binarize_otsu(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L"))
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))


def binarize_adaptive(image: Image.Image, block_size: int = 31, c: int = 10) -> Image.Image:
    gray = np.array(image.convert("L"))
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )
    return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
