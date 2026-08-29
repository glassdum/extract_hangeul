"""Bounding Box 간격 기반 정밀 띄어쓰기 (문서 "띄어쓰기 및 줄 연결").

문서: "문자 Bounding Box 간 거리를 평균 글자 폭과 비교해 실제 단어 간격을
계산한다. OCR 공백과 이미지 간격이 일치하면 공백을 확정한다."

PaddleOCR 검출기는 보통 한 줄/구를 박스 하나로 묶어 반환하므로, 이 프로젝트
구조에서 gap 분석이 실제로 의미 있는 단위는 "같은 행에서 인접한 두 검출
박스"다 — 그 사이 픽셀 간격이 두 박스의 평균 글자 폭보다 훨씬 좁으면,
검출기가 한 단어를 두 박스로 쪼갠 것으로 보고 공백 없이 이어 붙인다.
(PaddleOCR 인식기 내부의 문자 단위 컬럼 좌표(`return_word_box`)까지 내려가는
방법도 있지만, 이는 다운샘플링 비율 등 모델 구조에 따라 달라지는 비공개
구현 세부사항이라 신뢰할 수 있는 픽셀 좌표로 안전하게 변환할 근거가 없어
채택하지 않았다.)
"""

from __future__ import annotations

from common.types import TextLine

# 간격이 평균 글자 폭의 이 비율보다 좁으면 "쪼개진 한 단어"로 보고 공백 없이 붙인다.
DEFAULT_GLUE_GAP_RATIO = 0.3


def estimate_char_width(line: TextLine) -> float:
    """줄 하나의 평균 글자 폭 추정치. 빈 텍스트는 0을 반환한다."""
    length = len(line.text)
    if length == 0:
        return 0.0
    return line.bbox.width / length


def should_glue(left: TextLine, right: TextLine, gap_ratio: float = DEFAULT_GLUE_GAP_RATIO) -> bool:
    """같은 행에서 왼쪽->오른쪽으로 인접한 두 박스를 공백 없이 붙여야 하면 True.

    간격이 0 이하(겹치거나 맞닿음)면 무조건 붙인다. 판단 근거(글자 폭)가
    전혀 없으면 기존 동작대로 공백을 넣는 쪽(False)을 기본값으로 한다 —
    "원본에 없는 글자를 만들지 않는다"는 원칙과 마찬가지로, 근거 없이
    원본의 공백을 없애는 것도 피한다.
    """
    gap = right.bbox.x0 - left.bbox.x1
    if gap <= 0:
        return True

    widths = [w for w in (estimate_char_width(left), estimate_char_width(right)) if w > 0]
    if not widths:
        return False

    avg_char_width = sum(widths) / len(widths)
    return gap < avg_char_width * gap_ratio
