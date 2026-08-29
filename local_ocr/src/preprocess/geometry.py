"""기하 보정: Deskew (문서 "이미지 전처리" 표의 기하 보정 행 일부).

Perspective Transform(원근 보정)·문서 펴기는 아직 구현하지 않았다 — UVDoc
조건부 모델을 실제로 도입하는 시점(모델 학습/변환 단계 이후)에 함께 다루는
것이 낫다고 판단했다.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def estimate_skew_angle(gray: np.ndarray) -> float:
    """잉크 픽셀의 최소 회전 사각형에서 가장 긴 변의 기울기 각도(도)를 추정한다.

    `cv2.minAreaRect`가 반환하는 `angle` 자체는 OpenCV 버전마다 값 범위
    관례가 달라 그대로 쓰면 부호가 뒤집히거나 90도씩 어긋나는 경우가 있다.
    대신 `boxPoints`로 사각형 꼭짓점을 얻어 가장 긴 변(문서의 기준선/텍스트
    줄로 가정)의 각도를 수평 기준으로 직접 계산해 그 문제를 피한다.
    """
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 20:
        return 0.0

    box = cv2.boxPoints(cv2.minAreaRect(coords))
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    p1, p2 = max(edges, key=lambda edge: np.linalg.norm(edge[0] - edge[1]))

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # 텍스트 줄은 수평에 가까워야 하므로 [-45, 45] 범위로 접어 넣는다.
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return angle


def deskew(image: Image.Image, angle: float) -> Image.Image:
    array = np.array(image.convert("RGB"))
    height, width = array.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        array, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return Image.fromarray(rotated)
