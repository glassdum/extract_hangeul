from __future__ import annotations

import re
import unicodedata

from common.types import TextLine

_WHITESPACE_RE = re.compile(r"\s+")


def order_reading(lines: list[TextLine]) -> list[TextLine]:
    """줄들을 페이지 순서 -> 행(위→아래) -> 행 내 왼쪽→오른쪽 순으로 정렬한다."""
    ordered: list[TextLine] = []
    for page in sorted({line.page for line in lines}):
        page_lines = [line for line in lines if line.page == page]
        rows = _cluster_rows(page_lines)
        rows.sort(key=lambda row: min(line.bbox.y0 for line in row))
        for row in rows:
            row.sort(key=lambda line: line.bbox.x0)
            ordered.extend(row)
    return ordered


def join_text(lines: list[TextLine]) -> str:
    """읽기 순서로 정렬한 줄들을 공백 하나로 연결하고 NFC 정규화한다."""
    ordered = order_reading(lines)
    text = " ".join(line.text for line in ordered if line.text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return unicodedata.normalize("NFC", text)


def _cluster_rows(lines: list[TextLine]) -> list[list[TextLine]]:
    """수직 겹침을 기준으로 같은 행에 속하는 줄들을 묶는다 (간단한 읽기-순서 휴리스틱).

    여러 컬럼(다단 레이아웃)을 인식해 컬럼별로 먼저 읽는 것까지는 다루지
    않는다 — 단일/근사 단일 컬럼 문서를 전제로 한 Stage 1 근사치다.
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
