"""Shared dataclasses passed between pipeline stages.

`DocumentResult` is what `storage.writer` serializes to TXT/JSON, so its
shape follows the "결과 저장 형식" section of the plan (특히 JSON 예시의
text/status/page/bbox/candidates 필드).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 문서 "신뢰도와 판독 불가 정책" 표의 상태값. Stage 1은 단일 모델(Paddle)만
# 사용하므로 "판독 불가"·"재처리"는 아직 산출하지 않는다 (Stage 3에서 추가).
Status = Literal["auto_confirmed", "review_required", "low_confidence"]


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
    source: str  # "pdf_text" | "paddle_print" 등 (Stage 3에서 후보군으로 확장)
    status: Status = "auto_confirmed"


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
