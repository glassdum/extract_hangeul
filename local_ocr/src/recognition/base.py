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

    def recognize_batch(self, images: list[np.ndarray]) -> list[list[RecognizedItem]]:
        """여러 장을 한 번에 인식한다 (문서 "CPU 성능 최적화": "전체 페이지가
        아니라 검출된 Text Crop을 Batch로 인식한다").

        기본 구현은 그냥 한 장씩 반복 호출한다 — 모든 엔진이 실제 배치
        추론을 지원하지는 않기 때문이다(예: Tesseract 바이너리는 프로세스
        호출 단위가 이미지 한 장). 배치를 지원하는 엔진(`PaddleOCREngine`)만
        오버라이드해 진짜 한 번의 호출로 여러 장을 처리한다.
        """
        return [self.recognize(image) for image in images]
