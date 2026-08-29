"""Tesseract 5 LSTM 교차 판독 엔진 (문서 "사용 모델 구성"의 교차 판독 행).

"Tesseract는 기본 인식기가 아니라 Paddle 계열 결과와 독립적으로 비교하는
보조 엔진으로 사용한다" — 그래서 항상 실행하지 않고, PaddleOCR 결과가
불확실할 때만 `ensemble.cross_check`가 호출한다.

시스템에 Tesseract 5 바이너리와 kor/eng 언어 데이터가 설치돼 있어야 한다
(Windows 배포판에는 문서 "초기 Python 패키지 계획"대로 함께 번들한다).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import OCREngine, RecognizedItem


class TesseractEngine(OCREngine):
    def __init__(self, lang: str = "kor+eng", dpi: int = 300):
        import pytesseract  # 외부 바이너리 의존성이므로 실제 사용 시점에 import

        self._pytesseract = pytesseract
        self._lang = lang
        # Crop 하나 = 텍스트 한 줄이라고 가정하고 PSM 7(단일 줄)을 강제한다.
        # 기본 PSM(자동 페이지 분석)은 작은 낱줄 Crop에서 레이아웃 분석에
        # 실패해 글자를 통째로 잘못 읽는 경우가 많았다 (자체 확인).
        # DPI를 명시하지 않으면 작은 Crop에서 해상도를 잘못 추정해 같은
        # 문제가 재현되므로 항상 `--dpi`를 붙인다.
        self._config = f"--psm 7 --dpi {dpi}"

    def recognize(self, image: np.ndarray) -> list[RecognizedItem]:
        """Crop 하나에 보통 글자 한 덩어리만 있다고 가정하고, 검출된 단어들을
        한 줄로 합쳐 `RecognizedItem` 하나로 반환한다 (Paddle 결과와 1:1 비교 목적)."""
        pytesseract = self._pytesseract
        pil_image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB

        data = pytesseract.image_to_data(
            pil_image, lang=self._lang, config=self._config, output_type=pytesseract.Output.DICT
        )

        words: list[str] = []
        confidences: list[float] = []
        x0s: list[int] = []
        y0s: list[int] = []
        x1s: list[int] = []
        y1s: list[int] = []

        for i, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:  # conf == -1: 단어가 아닌 구조적 항목(줄/블록 등)
                continue
            words.append(text)
            confidences.append(conf)
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            x0s.append(x)
            y0s.append(y)
            x1s.append(x + w)
            y1s.append(y + h)

        if not words:
            return []

        text = " ".join(words)
        confidence = (sum(confidences) / len(confidences)) / 100.0  # Tesseract는 0~100 스케일
        polygon = [
            [min(x0s), min(y0s)],
            [max(x1s), min(y0s)],
            [max(x1s), max(y1s)],
            [min(x0s), max(y1s)],
        ]
        return [RecognizedItem(text=text, confidence=confidence, polygon=polygon)]
