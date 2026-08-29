"""PaddleOCR 통합 파이프라인 래퍼 (문서 "사용 모델 구성"의 기본 엔진).

PaddleOCR 3.x의 `PaddleOCR` 클래스는 문서 방향 분류(PP-LCNet doc_ori), 문서
펴기(UVDoc, 기본 비활성), 텍스트 검출(PP-OCRv5_mobile_det), 줄 방향 분류
(PP-LCNet textline_ori), 인식(korean_PP-OCRv5_mobile_rec)을 하나의 파이프라인
호출로 실행한다. 각 옵션은 `common.config.PipelineConfig`에서 결정된다.

모델 가중치는 최초 실행 시 PaddleOCR가 공식 저장소에서 자동 다운로드해
`~/.paddleocr`(또는 PADDLE_PDX_MODEL_SOURCE 설정 경로)에 캐시한다. 완전
오프라인 배포판(문서 "1차 완성 기준": 외부 서버와 통신하지 않고 모델 파일을
로컬에서만 로딩)을 만들려면 이 캐시를 `resources/models/`로 옮겨 넣고
로컬 경로를 가리키도록 설정하는 작업이 Stage 8(Windows 배포)에서 필요하다.
"""

from __future__ import annotations

import numpy as np

from common.config import PipelineConfig

from .base import OCREngine, RecognizedItem


class PaddleOCREngine(OCREngine):
    def __init__(self, config: PipelineConfig):
        # 무거운 의존성이므로 실제로 엔진을 쓸 때만 import한다.
        from paddleocr import PaddleOCR

        kwargs: dict[str, object] = dict(
            lang=config.lang,
            use_doc_orientation_classify=config.use_doc_orientation_classify,
            use_doc_unwarping=config.use_doc_unwarping,
            use_textline_orientation=config.use_textline_orientation,
            device="gpu" if config.use_gpu else "cpu",
        )
        if config.cpu_threads:
            kwargs["cpu_threads"] = config.cpu_threads

        self._ocr = PaddleOCR(**kwargs)

    def recognize(self, image: np.ndarray) -> list[RecognizedItem]:
        items: list[RecognizedItem] = []
        for res in self._ocr.predict(image):
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            polys = res.get("rec_polys")
            if polys is None:
                polys = res.get("rec_boxes") or []

            for text, score, poly in zip(texts, scores, polys):
                if not text:
                    continue
                items.append(
                    RecognizedItem(
                        text=text,
                        confidence=float(score),
                        polygon=_to_quad(poly),
                    )
                )
        return items


def _to_quad(poly) -> list[list[float]]:
    points = poly.tolist() if hasattr(poly, "tolist") else list(poly)
    if len(points) == 4 and not isinstance(points[0], (list, tuple)):
        x0, y0, x1, y1 = points
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return [[float(p[0]), float(p[1])] for p in points]
