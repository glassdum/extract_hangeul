"""불확실한 Crop에 시도해볼 전처리 보정본 생성 (문서 "전처리 프로파일" 산출물).

"전처리는 하나의 보정본을 강제로 선택하지 않는다. 원본과 보정본을 비교하고,
불확실한 Crop에만 다중 전처리를 적용한다." — 이 함수는 그 "다중 전처리"
후보군을 만든다. 실제로 각 보정본을 재인식해 최적 후보를 고르는 것은
`ensemble.reprocess`가 담당한다.

문서 "이미지 전처리" 표의 "적용 조건" 열을 저비용 판별기(`quality`)로
근사해, 해당하지 않는 보정은 아예 생성하지 않는다 (CPU 비용 절감).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .binarize import binarize_otsu
from .contrast import apply_clahe, stretch_contrast
from .denoise import denoise_bilateral
from .geometry import deskew, estimate_skew_angle
from .invert import invert
from .quality import is_likely_inverted, is_low_contrast
from .upscale import upscale

SKEW_ANGLE_THRESHOLD_DEG = 1.0


def generate_variants(image: Image.Image) -> dict[str, Image.Image]:
    gray = np.array(image.convert("L"))
    variants: dict[str, Image.Image] = {
        # 작은 글자 Crop은 검출 단계에서부터 자주 놓치므로 항상 시도한다.
        "upscale_2x": upscale(image, factor=2),
        # 배경이 복잡하거나 흐린 영역에 자주 도움이 되므로 항상 시도한다.
        "binarize_otsu": binarize_otsu(image),
        "denoise_bilateral": denoise_bilateral(image),
    }

    if is_low_contrast(gray):
        variants["clahe"] = apply_clahe(image)
        variants["contrast_stretch"] = stretch_contrast(image)

    angle = estimate_skew_angle(gray)
    if abs(angle) >= SKEW_ANGLE_THRESHOLD_DEG:
        variants["deskew"] = deskew(image, angle)

    if is_likely_inverted(gray):
        variants["invert"] = invert(image)

    return variants
