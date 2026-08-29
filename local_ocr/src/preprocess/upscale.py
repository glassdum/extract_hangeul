"""확대: 2배·4배 Lanczos/Cubic (문서 "이미지 전처리" 표, 작은 글자 Crop용)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_INTERPOLATIONS = {"lanczos": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC}


def upscale(image: Image.Image, factor: int = 2, interpolation: str = "lanczos") -> Image.Image:
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"지원하지 않는 보간 방식: {interpolation!r}")
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    resized = cv2.resize(
        array, (width * factor, height * factor), interpolation=_INTERPOLATIONS[interpolation]
    )
    return Image.fromarray(resized)
