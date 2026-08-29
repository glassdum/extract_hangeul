"""Shared dataclasses passed between pipeline stages.

`DocumentResult` is what `storage.writer` serializes to TXT/JSON, so its
shape follows the "결과 저장 형식" section of the plan (특히 JSON 예시의
text/status/page/bbox/candidates 필드).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 문서 "신뢰도와 판독 불가 정책" 표의 상태값.
# - auto_confirmed: Stage 1 단일 모델 고신뢰 결과, 또는 Stage 3에서 Paddle과
#   Tesseract가 완전히 일치해 교차 확정된 결과.
# - review_required: 후보가 2개 이상이고 근거가 비슷해 사람 확인이 필요한 경우.
# - low_confidence: 재판독·교차 판독으로도 더 확신을 높이지 못한 단일 결과.
# - unreadable: Paddle·Tesseract 결과 일부가 끝내 일치하지 않아 그 구간만
#   UNREADABLE_MARKER로 대체된 경우 ("잘림·번짐·겹침" 정책에 대응).
Status = Literal["auto_confirmed", "review_required", "low_confidence", "unreadable"]

# 문서 "신뢰도와 판독 불가 정책" 예시("홍[판독 불가]동")에 쓰인 표시.
UNREADABLE_MARKER = "[판독 불가]"


@dataclass(frozen=True)
class BBox:
    """축 정렬 바운딩 박스 (이미지/페이지 픽셀 좌표)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @classmethod
    def from_polygon(cls, points: list[list[float]] | list[tuple[float, float]]) -> "BBox":
        """PaddleOCR 등이 반환하는 4점(quad) 폴리곤에서 축 정렬 박스를 만든다."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class TextLine:
    """검출·인식된 한 줄(또는 텍스트 영역)."""

    page: int
    bbox: BBox
    text: str
    confidence: float
    source: str  # "pdf_text" | "paddle_print" | "paddle_print_<variant>" 등
    status: Status = "auto_confirmed"
    # Stage 2 재판독을 거친 경우, 시도한 모든 후보 {candidate_key: (text, confidence)}.
    # 재판독하지 않은 줄은 비어 있다 — storage.writer가 그 경우 source 하나로 채운다.
    candidates: dict[str, tuple[str, float]] = field(default_factory=dict)


@dataclass
class PageResult:
    page: int
    width: int
    height: int
    lines: list[TextLine] = field(default_factory=list)


@dataclass
class DocumentResult:
    source_path: str
    pages: list[PageResult] = field(default_factory=list)
    final_text: str = ""

    def all_lines(self) -> list[TextLine]:
        return [line for page in self.pages for line in page.lines]
