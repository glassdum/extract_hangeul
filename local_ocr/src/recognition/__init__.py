"""인쇄체·손글씨·Tesseract 인식 엔진.

기본 인식은 PaddleOCR(`PaddleOCREngine`), 교차 판독은 Tesseract
(`TesseractEngine`)를 쓴다. 손글씨 Fine-tuning 엔진(Stage 4)은 아직 없다 —
공개 모델만으로는 목표 정확도를 보장하기 어려워 별도 학습이 필요하다
(문서 "한국어 손글씨 모델 개발" 참고). 세 엔진 모두 같은 `OCREngine`
인터페이스를 구현해 `ensemble`이 결과를 동일하게 다룰 수 있다.
"""

from .base import OCREngine, RecognizedItem
from .paddle_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine

__all__ = ["OCREngine", "RecognizedItem", "PaddleOCREngine", "TesseractEngine"]
