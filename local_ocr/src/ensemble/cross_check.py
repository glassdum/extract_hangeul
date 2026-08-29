"""Stage 3: PaddleOCR와 Tesseract의 독립 결과를 비교해 최종 판정을 내린다.

문서 "신뢰도와 판독 불가 정책"을 지금 가진 두 엔진(Paddle 인쇄체,
Tesseract)만으로 근사한다. 손글씨 Fine-tuning 모델(Stage 4 전까지 없음)이
빠져 있어 "복수 모델 일치"는 Paddle-Tesseract 일치로 대신한다.

두 결과가 전혀 다르면 글자 단위로 억지로 짜깁기하지 않고 "사용자 확인"으로
넘긴다 — 원본에 없는 글자를 만들어내는 위험을 피하기 위해서다 (문서
"1차 제외 범위": "원본에 보이지 않는 글자를 문맥만으로 생성하는 자동 보완").
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass

from common.types import UNREADABLE_MARKER, Status

# 이보다 낮은 SequenceMatcher.ratio()는 "같은 글자를 다르게 읽은 것"이 아니라
# "서로 다른 후보"로 본다 — 이때는 짜깁기 대신 사용자 확인으로 넘긴다.
MERGE_SIMILARITY_THRESHOLD = 0.5


@dataclass
class CrossCheckResult:
    text: str
    status: Status


def cross_check_texts(primary_text: str, secondary_text: str) -> CrossCheckResult:
    """primary(Paddle 최적 후보)와 secondary(Tesseract) 결과를 비교한다."""
    a = unicodedata.normalize("NFC", primary_text).strip()
    b = unicodedata.normalize("NFC", secondary_text).strip()

    if a == b:
        return CrossCheckResult(text=a, status="auto_confirmed")

    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    if ratio < MERGE_SIMILARITY_THRESHOLD:
        return CrossCheckResult(text=primary_text, status="review_required")

    merged, had_mismatch = merge_with_markers(a, b)
    status: Status = "unreadable" if had_mismatch else "auto_confirmed"
    return CrossCheckResult(text=merged, status=status)


def merge_with_markers(a: str, b: str, marker: str = UNREADABLE_MARKER) -> tuple[str, bool]:
    """일치하는 구간은 그대로 두고, 불일치하는 구간만 marker 하나로 접는다.

    예: merge_with_markers("홍길동", "홍김동") -> ("홍[판독 불가]동", True)
    """
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts: list[str] = []
    had_mismatch = False

    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(a[i1:i2])
            continue
        had_mismatch = True
        if parts and parts[-1] == marker:
            continue  # 연속된 불일치 구간은 marker 하나로 합친다
        parts.append(marker)

    return "".join(parts), had_mismatch
