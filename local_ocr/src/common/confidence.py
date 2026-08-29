"""단일 엔진 confidence만으로 상태를 근사 판정한다.

문서 "신뢰도와 판독 불가 정책"의 진짜 판정 기준(복수 모델 일치 여부, 공간
검증)은 여러 `OCREngine` 결과를 비교하는 `ensemble`(Stage 3)이 담당한다.
Stage 1은 아직 단일 엔진(PaddleOCR)만 실행하므로, 그 결과가 나오기 전까지
쓸 수 있는 최소한의 근사치로 confidence 임계값만 사용한다. "판독 불가"는
교차 검증 없이는 내리지 않는다 — 낮은 confidence는 `low_confidence`로만
표시해 사용자가 원본 대비 검토할 수 있게 남긴다.
"""

from __future__ import annotations

from common.config import PipelineConfig
from common.types import Status


def classify_status(confidence: float, config: PipelineConfig) -> Status:
    if confidence >= config.auto_confirm_threshold:
        return "auto_confirmed"
    if confidence >= config.review_threshold:
        return "review_required"
    return "low_confidence"
