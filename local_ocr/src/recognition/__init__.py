"""인쇄체·손글씨·Tesseract 인식 엔진.

Stage 1은 PaddleOCR 하나(`PaddleOCREngine`)만 사용한다. 손글씨 Fine-tuning
엔진(Stage 4)과 Tesseract 교차 판독 엔진(Stage 3)은 동일한 `OCREngine`
인터페이스를 구현하는 형태로 이후 추가된다.
"""

from .base import OCREngine, RecognizedItem
from .paddle_engine import PaddleOCREngine

__all__ = ["OCREngine", "RecognizedItem", "PaddleOCREngine"]
