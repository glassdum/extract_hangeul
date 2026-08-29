"""후보 비교·신뢰도 판정.

Stage 2: `reprocess`가 불확실한 Crop을 여러 전처리 보정본으로 재인식해
같은 엔진(PaddleOCR) 안에서 최적 후보를 고른다.

Stage 3(예정): 인쇄체·손글씨 Fine-tuning·Tesseract처럼 서로 다른 엔진의
결과까지 함께 비교해 문서 "신뢰도와 판독 불가 정책"의 자동 확정/재처리/
사용자 확인/판독 불가를 판정하는 로직이 이 패키지에 추가된다. 지금은
`common.confidence.classify_status`의 confidence 임계값이 그 근사치다.
"""

from .reprocess import VariantCandidate, pick_best, reprocess_crop

__all__ = ["VariantCandidate", "pick_best", "reprocess_crop"]
