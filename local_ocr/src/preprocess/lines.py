"""선 제거: Morphology 기반 가로·세로선 제거 (문서 "이미지 전처리" 표).

표 선이 글자를 방해할 때만 쓰라고 명시돼 있고 자동 감지가 쉽지 않으므로,
`variants.generate_variants()`의 기본 후보에는 포함하지 않았다. 표 안의
텍스트로 확인된 영역에서만 호출하는 용도로 함수만 제공한다.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def remove_table_lines(image: Image.Image, min_line_length_ratio: float = 0.5) -> Image.Image:
    gray = np.array(image.convert("L"))
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(int(width * min_line_length_ratio), 1), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(int(height * min_line_length_ratio), 1))
    )

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)

    array = np.array(image.convert("RGB"))
    array[lines_mask > 0] = (255, 255, 255)
    return Image.fromarray(array)
