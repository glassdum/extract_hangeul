"""노이즈 제거: Median/Bilateral Denoising (문서 "이미지 전처리" 표)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def denoise_median(image: Image.Image, ksize: int = 3) -> Image.Image:
    array = np.array(image.convert("RGB"))
    return Image.fromarray(cv2.medianBlur(array, ksize))


def denoise_bilateral(image: Image.Image) -> Image.Image:
    array = np.array(image.convert("RGB"))
    filtered = cv2.bilateralFilter(array, d=9, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(filtered)
