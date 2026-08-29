from __future__ import annotations

import re
import unicodedata

from common.types import TextLine

from .gap_analysis import should_glue

_WHITESPACE_RE = re.compile(r"\s+")


def order_reading(lines: list[TextLine]) -> list[TextLine]:
    """줄들을 페이지 순서 -> 행(위→아래) -> 행 내 왼쪽→오른쪽 순으로 정렬한다."""
    return [line for row in _rows_in_reading_order(lines) for line in row]


def join_text(lines: list[TextLine]) -> str:
    """읽기 순서로 정렬한 줄들을 연결하고 NFC 정규화한다.

    같은 행 안에서 인접한 두 박스는 Bounding Box 간격을 평균 글자 폭과
    비교해 붙일지(같은 단어가 쪼개진 경우) 공백을 넣을지(원래 다른 단어)
    판정한다 (`spacing.gap_analysis`). 행이 바뀔 때는 이 판정과 무관하게
    항상 공백 하나로 연결한다 (줄바꿈 -> 공백 하나 변환).
    """
    row_texts = [_join_row(row) for row in _rows_in_reading_order(lines)]
    text = " ".join(part for part in row_texts if part)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return unicodedata.normalize("NFC", text)


def _join_row(row: list[TextLine]) -> str:
    parts: list[str] = []
    prev: TextLine | None = None
    for line in row:
        if not line.text:
            continue
        if prev is not None and not should_glue(prev, line):
            parts.append(" ")
        parts.append(line.text)
        prev = line
    return "".join(parts)


def _rows_in_reading_order(lines: list[TextLine]) -> list[list[TextLine]]:
    ordered_rows: list[list[TextLine]] = []
    for page in sorted({line.page for line in lines}):
        page_lines = [line for line in lines if line.page == page]
        rows = _cluster_rows(page_lines)
        rows.sort(key=lambda row: min(line.bbox.y0 for line in row))
        for row in rows:
            row.sort(key=lambda line: line.bbox.x0)
        ordered_rows.extend(rows)
    return ordered_rows


def _cluster_rows(lines: list[TextLine]) -> list[list[TextLine]]:
    """수직 겹침을 기준으로 같은 행에 속하는 줄들을 묶는다 (간단한 읽기-순서 휴리스틱).

    여러 컬럼(다단 레이아웃)을 인식해 컬럼별로 먼저 읽는 것까지는 다루지
    않는다 — 단일/근사 단일 컬럼 문서를 전제로 한 근사치다.
    """
    rows: list[list[TextLine]] = []
    row_ranges: list[tuple[float, float]] = []

    for line in sorted(lines, key=lambda l: l.bbox.y0):
        placed = False
        for i, (row_y0, row_y1) in enumerate(row_ranges):
            overlap = min(line.bbox.y1, row_y1) - max(line.bbox.y0, row_y0)
            min_height = min(line.bbox.height, row_y1 - row_y0) or 1.0
            if overlap > 0.5 * min_height:
                rows[i].append(line)
                row_ranges[i] = (min(row_y0, line.bbox.y0), max(row_y1, line.bbox.y1))
                placed = True
                break
        if not placed:
            rows.append([line])
            row_ranges.append((line.bbox.y0, line.bbox.y1))

    return rows
