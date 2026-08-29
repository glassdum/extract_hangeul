"""인쇄체·손글씨·Tesseract 인식 엔진.

기본 인식은 PaddleOCR(`PaddleOCREngine`), 교차 판독은 Tesseract
(`TesseractEngine`)를 쓴다. `HandwritingEngine`은 손글씨 Fine-tuning
가중치(문서 "한국어 손글씨 모델 개발")를 위한 자리다 — 실제 학습된 가중치는
아직 없고(`resources/handwriting_training/README.md` 참고), 없는 상태로
생성하면 `HandwritingModelNotAvailableError`를 던진다. 세 엔진 모두 같은
`OCREngine` 인터페이스를 구현해 `ensemble`이 결과를 동일하게 다룰 수 있다.

`CachingEngine`은 어떤 `OCREngine`이든 감싸 결과를 해시 기반으로 캐싱한다
(문서 "CPU 성능 최적화").
"""

from .base import OCREngine, RecognizedItem
from .caching_engine import CachingEngine
from .handwriting_engine import HandwritingEngine, HandwritingModelNotAvailableError
from .paddle_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine

__all__ = [
    "OCREngine",
    "RecognizedItem",
    "PaddleOCREngine",
    "TesseractEngine",
    "HandwritingEngine",
    "HandwritingModelNotAvailableError",
    "CachingEngine",
]
