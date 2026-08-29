"""입력 파일 형식 판별 및 PDF/이미지 로더 통합 진입점."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from common.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_PDF_EXTENSIONS
from common.types import BBox

from .image_loader import load_image_frames
from .pdf_loader import LoadedPage, OcrRegion, load_pdf

InputType = Literal["pdf", "image"]


def detect_input_type(path: str | Path) -> InputType:
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix or path}")


def load_document(path: str | Path, dpi: int, min_text_layer_chars: int) -> list[LoadedPage]:
    """PDF와 이미지를 동일한 `LoadedPage` 목록으로 통일해 반환한다.

    - PDF: 페이지별 직접 추출 텍스트(text_lines) + OCR 대상 이미지(ocr_regions).
    - 이미지: text_lines는 항상 비어있고, 프레임(다중 페이지 TIFF/GIF 포함) 전체가
      단일 ocr_region이 된다.
    """
    kind = detect_input_type(path)

    if kind == "pdf":
        return load_pdf(path, dpi=dpi, min_text_layer_chars=min_text_layer_chars)

    frames = load_image_frames(path)
    pages: list[LoadedPage] = []
    for frame in frames:
        width, height = frame.image.size
        pages.append(
            LoadedPage(
                page=frame.index,
                width=width,
                height=height,
                text_lines=[],
                ocr_regions=[OcrRegion(bbox=BBox(0, 0, width, height), image=frame.image)],
            )
        )
    return pages


__all__ = ["detect_input_type", "load_document", "LoadedPage", "OcrRegion"]
