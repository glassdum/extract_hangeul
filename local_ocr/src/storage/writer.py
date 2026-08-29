"""문서 "결과 저장 형식" 구현: TXT(최종 문자열만) / JSON(페이지·좌표·후보·상태)."""

from __future__ import annotations

import json
from pathlib import Path

from common.types import DocumentResult, TextLine


def save_txt(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def save_json(path: str | Path, doc: DocumentResult) -> None:
    payload = {
        "source": doc.source_path,
        "final_text": doc.final_text,
        "pages": [
            {
                "page": page.page,
                "width": page.width,
                "height": page.height,
                "lines": [_line_to_dict(line) for line in page.lines],
            }
            for page in doc.pages
        ],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _line_to_dict(line: TextLine) -> dict:
    return {
        "text": line.text,
        "status": line.status,
        "page": line.page,
        "bbox": line.bbox.as_list(),
        "candidates": {
            line.source: [line.text, round(line.confidence, 4)],
        },
    }
