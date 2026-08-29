"""OCR 엔진 공통 인터페이스.

문서 "사용 모델 구성"의 여러 엔진(기본 인식/손글씨 인식/교차 판독)을 같은
형태로 다루기 위한 최소 계약. Ensemble(Stage 3)은 여러 `OCREngine` 구현체의
`recognize()` 결과를 비교해 최종 후보를 정한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RecognizedItem:
    """검출+인식 결과 한 건. polygon은 입력 이미지 로컬 픽셀 좌표의 4점(quad)."""

    text: str
    confidence: float
    polygon: list[list[float]]


class OCREngine(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> list[RecognizedItem]:
        """BGR ndarray 한 장을 검출+인식해 텍스트 항목 목록을 반환한다."""
        raise NotImplementedError
