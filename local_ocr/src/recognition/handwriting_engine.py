"""한국어 손글씨 Fine-tuning 인식 엔진 (문서 "한국어 손글씨 모델 개발").

문서: "korean_PP-OCRv5_mobile_rec_handwriting_ft는 공개 모델의 공식 이름이
아니라 본 프로젝트에서 만들어야 하는 손글씨 Fine-tuning 결과물의 내부
이름이다." 이 저장소를 개발한 환경(코드 작성 전용 샌드박스 — GPU 없음,
AI Hub 등 외부 데이터셋 다운로드가 네트워크 정책상 차단됨)에서는 실제
Fine-tuning을 실행할 수 없다. 학습 자체는 `resources/handwriting_training/`
의 스크립트로 GPU가 있는 별도 환경에서 사용자가 직접 수행해야 한다.

이 클래스는 그렇게 나온 가중치를 바로 쓸 수 있도록 인터페이스만 미리
맞춰 둔 것이다. `paddleocr.TextRecognition`을 감지·인식 전체 파이프라인이
아니라 인식기만 단독으로 쓰는 이유는, 이 엔진이 받는 입력이 항상 이미
(1차 검출 또는 Stage 2 재크롭으로) 한 줄로 잘린 Crop이기 때문이다 — 문서
"모델 선택 근거": "동일한 한국어 사전과 경량 구조를 유지"하므로 검출기는
기본 인식기(`PaddleOCREngine`)와 공유하고 인식기만 교체한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import OCREngine, RecognizedItem

# PaddleOCR/PaddleX가 추론 모델 디렉터리에 남기는 대표 파일 확장자들.
_MODEL_FILE_GLOBS = ("*.pdiparams", "*.pdmodel", "*.onnx", "inference.*")


class HandwritingModelNotAvailableError(RuntimeError):
    """학습된 손글씨 Fine-tuning 가중치가 아직 없을 때 발생한다."""


class HandwritingEngine(OCREngine):
    def __init__(self, model_dir: str | Path, lang: str = "korean"):
        model_dir = Path(model_dir)
        if not _has_model_files(model_dir):
            raise HandwritingModelNotAvailableError(
                f"손글씨 Fine-tuning 모델을 찾을 수 없습니다: {model_dir}\n"
                "resources/handwriting_training/README.md의 절차대로 학습한 뒤, "
                "그 결과물(추론 모델 디렉터리)을 이 경로에 배치하세요."
            )

        from paddleocr import TextRecognition  # 무거운 의존성이므로 실제 사용 시점에 import

        self._recognizer = TextRecognition(
            # 손글씨 FT도 인쇄체 기본 모델과 같은 구조/사전을 유지하므로 같은
            # model_name을 쓰고, 가중치만 Fine-tuning 결과물(model_dir)로 바꾼다.
            model_name="korean_PP-OCRv5_mobile_rec",
            model_dir=str(model_dir),
        )

    def recognize(self, image: np.ndarray) -> list[RecognizedItem]:
        results = self._recognizer.predict(image)
        if not results:
            return []

        result = results[0]
        text = result.get("rec_text", "")
        if not text:
            return []
        confidence = float(result.get("rec_score", 0.0))

        height, width = image.shape[:2]
        polygon = [[0, 0], [width, 0], [width, height], [0, height]]
        return [RecognizedItem(text=text, confidence=confidence, polygon=polygon)]


def _has_model_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return any(any(model_dir.glob(pattern)) for pattern in _MODEL_FILE_GLOBS)
