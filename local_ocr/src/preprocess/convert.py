"""PIL <-> ndarray 변환. PaddleOCR/OpenCV 계열 API는 BGR ndarray를 기대한다."""

from __future__ import annotations

import numpy as np
from PIL import Image


def to_bgr_ndarray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])
