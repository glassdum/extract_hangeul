"""Stage 2: 불확실한 Crop을 여러 전처리 보정본으로 재인식해 최적 후보를 고른다.

문서 "인쇄체·손글씨 인식 전략": "결과가 다르면 여러 전처리본을 만들고
Tesseract를 추가 실행한다"의 앞부분(전처리본 비교)을 다룬다. 여기서 고른
최적 후보의 이미지는 Stage 3(`ensemble.cross_check`)이 그대로 재사용해
Tesseract로도 교차 판독한다 — 전처리 보정본을 두 번 만들지 않기 위해
`VariantCandidate`가 이미지 자체도 들고 있다.

문서 "CPU 성능 최적화": "전체 페이지가 아니라 검출된 Text Crop을 Batch로
인식한다." — 원본 Crop + 모든 보정본을 `OCREngine.recognize_batch()`
한 번으로 처리한다(엔진이 진짜 배치를 지원하면 `PaddleOCREngine`처럼 모델
호출이 한 번으로 줄고, 지원하지 않으면 기본 구현이 순회할 뿐이라 손해는
없다).
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from preprocess.convert import to_bgr_ndarray
from preprocess.variants import generate_variants
from recognition.base import OCREngine, RecognizedItem


@dataclass
class VariantCandidate:
    variant: str  # "original" 또는 preprocess.variants가 붙인 이름
    image: Image.Image  # 이 변형이 실제로 인식을 시도한 이미지
    item: RecognizedItem | None  # 해당 보정본에서 아무 글자도 검출되지 않으면 None


def reprocess_crop(crop: Image.Image, engine: OCREngine) -> list[VariantCandidate]:
    """원본 Crop과 모든 전처리 보정본을 같은 엔진으로, 한 번의 배치 호출로 재인식한다."""
    variants = generate_variants(crop)
    names = ["original", *variants.keys()]
    images = [crop, *variants.values()]

    batch_results = engine.recognize_batch([to_bgr_ndarray(image) for image in images])

    return [
        VariantCandidate(variant=name, image=image, item=_pick_best(items))
        for name, image, items in zip(names, images, batch_results)
    ]


def pick_best(candidates: list[VariantCandidate]) -> VariantCandidate | None:
    scored = [c for c in candidates if c.item is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c.item.confidence)


def best_item(engine: OCREngine, image: Image.Image) -> RecognizedItem | None:
    """Crop 하나에는 보통 글자 한 덩어리만 있어야 하므로, 검출된 항목 중 confidence가
    가장 높은 하나만 대표값으로 쓴다. (단일 이미지용 — 배치 경로는 `reprocess_crop` 참고.)"""
    return _pick_best(engine.recognize(to_bgr_ndarray(image)))


def _pick_best(items: list[RecognizedItem]) -> RecognizedItem | None:
    if not items:
        return None
    return max(items, key=lambda item: item.confidence)
