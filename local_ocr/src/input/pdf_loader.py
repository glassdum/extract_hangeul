"""PDF 로딩: 텍스트 레이어 판별, 직접 추출, 혼합/스캔 페이지 렌더링.

문서 "처리 파이프라인": "PDF가 실제 텍스트를 포함하는지 검사하고, 신뢰할 수
있는 텍스트는 직접 추출한다" 및 "파일 유형별 입력 처리" 표(텍스트/스캔/혼합
PDF)를 페이지 단위로 구현한다.

페이지에 텍스트 레이어가 있으면 단어 단위로 직접 추출하고, 그 페이지에
포함된 래스터 이미지(임베디드 이미지)만 잘라내 OCR 대상으로 넘긴다. 텍스트
레이어가 전혀 없으면 스캔 페이지로 보고 페이지 전체를 렌더링해 OCR한다.

주의: 텍스트/이미지가 같은 픽셀 영역에서 겹치는 복잡한 레이아웃 분리나,
저품질 텍스트 레이어 감지(예: OCR로 생성된 잘못된 텍스트 레이어)는 다루지
않는다 — Stage 2 이후 범위.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (>=1.24 renamed the module; `fitz` alias is deprecated)
from PIL import Image

from common.types import BBox, TextLine


@dataclass
class OcrRegion:
    """OCR 엔진에 넘길 이미지 조각과, 그 결과를 다시 페이지 좌표로 되돌리기 위한 bbox."""

    bbox: BBox
    image: Image.Image


@dataclass
class LoadedPage:
    page: int  # 1-based
    width: int
    height: int
    text_lines: list[TextLine] = field(default_factory=list)
    ocr_regions: list[OcrRegion] = field(default_factory=list)


def _pixmap_to_pil(pix: "fitz.Pixmap") -> Image.Image:
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def load_pdf(path: str | Path, dpi: int, min_text_layer_chars: int) -> list[LoadedPage]:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[LoadedPage] = []

    with fitz.open(str(path)) as doc:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_no = page_index + 1
            raw_text = page.get_text("text")
            has_text_layer = len(raw_text.strip()) >= min_text_layer_chars

            width = round(page.rect.width * zoom)
            height = round(page.rect.height * zoom)

            text_lines: list[TextLine] = []
            ocr_regions: list[OcrRegion] = []

            if has_text_layer:
                text_lines = _extract_text_lines(page, page_no, zoom)
                ocr_regions = _extract_embedded_image_regions(page, matrix, zoom)
            else:
                pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
                image = _pixmap_to_pil(pix)
                ocr_regions.append(OcrRegion(bbox=BBox(0, 0, pix.width, pix.height), image=image))
                # 렌더링된 실제 픽셀 크기를 페이지 크기로 사용한다 (반올림 오차로
                # round(page.rect * zoom)과 pix.width/height가 1px 어긋날 수 있음).
                width, height = pix.width, pix.height

            pages.append(
                LoadedPage(
                    page=page_no,
                    width=width,
                    height=height,
                    text_lines=text_lines,
                    ocr_regions=ocr_regions,
                )
            )

    return pages


def _extract_text_lines(page: "fitz.Page", page_no: int, zoom: float) -> list[TextLine]:
    """페이지 내장 텍스트 레이어에서 줄 단위 텍스트를 직접 추출한다.

    PyMuPDF의 "words" 추출은 (x0,y0,x1,y1,word,block_no,line_no,word_no)를
    반환한다. block/line 번호로 묶어 원본 줄 구조를 복원한다.
    """
    words = page.get_text("words")
    lines_map: dict[tuple[int, int], list[tuple[float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, word, block_no, line_no, _word_no in words:
        lines_map.setdefault((block_no, line_no), []).append((x0, y0, x1, y1, word))

    text_lines: list[TextLine] = []
    for word_list in lines_map.values():
        word_list.sort(key=lambda w: w[0])
        line_text = " ".join(w[4] for w in word_list)
        x0 = min(w[0] for w in word_list) * zoom
        y0 = min(w[1] for w in word_list) * zoom
        x1 = max(w[2] for w in word_list) * zoom
        y1 = max(w[3] for w in word_list) * zoom
        text_lines.append(
            TextLine(
                page=page_no,
                bbox=BBox(x0, y0, x1, y1),
                text=line_text,
                confidence=1.0,
                source="pdf_text",
                status="auto_confirmed",
            )
        )
    return text_lines


def _extract_embedded_image_regions(
    page: "fitz.Page", matrix: "fitz.Matrix", zoom: float
) -> list[OcrRegion]:
    """혼합 PDF: 텍스트 레이어가 있는 페이지에 포함된 래스터 이미지만 OCR 대상으로 잘라낸다."""
    regions: list[OcrRegion] = []
    seen_rects: set[tuple[float, float, float, float]] = set()

    for img in page.get_images(full=True):
        xref = img[0]
        for rect in page.get_image_rects(xref):
            key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
            if key in seen_rects:
                continue  # 동일 이미지가 여러 위치에 배치된 경우 중복 방지
            seen_rects.add(key)

            pix = page.get_pixmap(matrix=matrix, clip=rect, colorspace=fitz.csRGB, alpha=False)
            if pix.width == 0 or pix.height == 0:
                continue
            image = _pixmap_to_pil(pix)
            bbox = BBox(rect.x0 * zoom, rect.y0 * zoom, rect.x1 * zoom, rect.y1 * zoom)
            regions.append(OcrRegion(bbox=bbox, image=image))

    return regions
