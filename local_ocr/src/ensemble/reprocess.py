"""Stage 2: 불확실한 Crop을 여러 전처리 보정본으로 재인식해 최적 후보를 고른다.

문서 "인쇄체·손글씨 인식 전략": "결과가 다르면 여러 전처리본을 만들고
Tesseract를 추가 실행한다"의 앞부분(전처리본 비교)을 다룬다. 여기서 고른
최적 후보의 이미지는 Stage 3(`ensemble.cross_check`)이 그대로 재사용해
Tesseract로도 교차 판독한다 — 전처리 보정본을 두 번 만들지 않기 위해
`VariantCandidate`가 이미지 자체도 들고 있다.
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
    """원본 Crop과 모든 전처리 보정본을 같은 엔진으로 재인식한다."""
    candidates = [
        VariantCandidate(variant="original", image=crop, item=best_item(engine, crop))
    ]
    for name, variant_image in generate_variants(crop).items():
        candidates.append(
            VariantCandidate(
                variant=name, image=variant_image, item=best_item(engine, variant_image)
            )
        )
    return candidates


def pick_best(candidates: list[VariantCandidate]) -> VariantCandidate | None:
    scored = [c for c in candidates if c.item is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c.item.confidence)


def best_item(engine: OCREngine, image: Image.Image) -> RecognizedItem | None:
    """Crop 하나에는 보통 글자 한 덩어리만 있어야 하므로, 검출된 항목 중 confidence가
    가장 높은 하나만 대표값으로 쓴다."""
    items = engine.recognize(to_bgr_ndarray(image))
    if not items:
        return None
    return max(items, key=lambda item: item.confidence)
