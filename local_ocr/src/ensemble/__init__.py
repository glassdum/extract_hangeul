"""후보 비교·신뢰도 판정.

Stage 2: `reprocess`가 불확실한 Crop을 여러 전처리 보정본으로 재인식해
같은 엔진(PaddleOCR) 안에서 최적 후보를 고른다.

Stage 3: `cross_check`가 그 최적 후보를 Tesseract 결과와 비교해 문서
"신뢰도와 판독 불가 정책"의 자동 확정/사용자 확인/판독 불가를 판정한다.
손글씨 Fine-tuning 모델(Stage 4)이 아직 없어 "복수 모델 일치"는
Paddle-Tesseract 일치로 근사한다.
"""

from .cross_check import CrossCheckResult, cross_check_texts, merge_with_markers
from .reprocess import VariantCandidate, best_item, pick_best, reprocess_crop

__all__ = [
    "VariantCandidate",
    "best_item",
    "pick_best",
    "reprocess_crop",
    "CrossCheckResult",
    "cross_check_texts",
    "merge_with_markers",
]
