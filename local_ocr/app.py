#!/usr/bin/env python3
"""CPU 기반 완전 로컬 OCR 프로그램 - Stage 1(입력·기본 OCR) CLI 진입점.

파일 로딩, PDF 텍스트 레이어 판별, 문서/줄 방향 보정(PaddleOCR 내장),
기본 한국어 인식 모델 실행까지 수행해 TXT·JSON 결과를 생성한다.
손글씨 Fine-tuning, Tesseract 교차 판독, 검토 GUI, Windows 패키징은
이후 단계(Stage 3~8)에서 추가된다.

Usage:
    python app.py <input-file-or-dir> [--output-dir output] [--dpi 300]
                  [--mode accuracy|speed] [--lang korean]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from common.config import (  # noqa: E402
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    PipelineConfig,
)
from common.confidence import classify_status  # noqa: E402
from common.types import BBox, DocumentResult, PageResult, TextLine  # noqa: E402
from ensemble.reprocess import pick_best, reprocess_crop  # noqa: E402
from input.loader import load_document  # noqa: E402
from preprocess.convert import to_bgr_ndarray  # noqa: E402
from preprocess.crop import crop_with_padding  # noqa: E402
from recognition.paddle_engine import PaddleOCREngine  # noqa: E402
from spacing.reading_order import join_text  # noqa: E402
from storage.writer import save_json, save_txt  # noqa: E402

SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


def run_pipeline(
    path: str | Path, config: PipelineConfig, engine: PaddleOCREngine
) -> DocumentResult:
    """문서 "처리 파이프라인" 1~2, 5~9, 11~12번 항목: 로딩, 기본 인식, 불확실한
    Crop 재판독(Stage 2)부터 TXT/JSON용 DocumentResult 구성까지. (3~4번 기하
    보정 조건부 적용, 10번 판독 불가/공간 검증은 Stage 3~5에서 추가된다.)
    """
    loaded_pages = load_document(
        path, dpi=config.dpi, min_text_layer_chars=config.min_text_layer_chars
    )

    pages: list[PageResult] = []
    for loaded in loaded_pages:
        # PDF 텍스트 레이어는 이미 신뢰할 수 있는 문자열이므로 그대로 사용한다.
        lines: list[TextLine] = list(loaded.text_lines)

        for region in loaded.ocr_regions:
            ndarray = to_bgr_ndarray(region.image)
            for item in engine.recognize(ndarray):
                local_bbox = BBox.from_polygon(item.polygon)
                page_bbox = BBox(
                    region.bbox.x0 + local_bbox.x0,
                    region.bbox.y0 + local_bbox.y0,
                    region.bbox.x0 + local_bbox.x1,
                    region.bbox.y0 + local_bbox.y1,
                )
                lines.append(
                    _build_text_line(
                        page=loaded.page,
                        page_bbox=page_bbox,
                        local_bbox=local_bbox,
                        region_image=region.image,
                        text=item.text,
                        confidence=item.confidence,
                        engine=engine,
                        config=config,
                    )
                )

        pages.append(
            PageResult(page=loaded.page, width=loaded.width, height=loaded.height, lines=lines)
        )

    doc = DocumentResult(source_path=str(path), pages=pages)
    doc.final_text = join_text(doc.all_lines())
    return doc


def _build_text_line(
    *,
    page: int,
    page_bbox: BBox,
    local_bbox: BBox,
    region_image: Image.Image,
    text: str,
    confidence: float,
    engine: PaddleOCREngine,
    config: PipelineConfig,
) -> TextLine:
    """1차 인식 결과로 TextLine을 만들되, confidence가 낮으면 Stage 2 재판독을 거친다.

    Stage 2 재판독은 1차 인식이 본 전체 영역이 아니라 그 안의 좁은 Crop만 다시
    보므로, `reprocess_crop`의 "original"(보정 없는 Crop) 결과조차 1차 인식과
    다른 입력에 대한 새로운 시도다 — 그래서 1차 인식값을 덮어쓰지 않고 별도
    후보(`paddle_print_crop`)로 남긴다.
    """
    base_source = "paddle_print"
    status = classify_status(confidence, config)

    if status == "auto_confirmed":
        return TextLine(
            page=page,
            bbox=page_bbox,
            text=text,
            confidence=confidence,
            source=base_source,
            status=status,
        )

    crop = crop_with_padding(region_image, local_bbox)
    candidates_list = reprocess_crop(crop, engine)

    candidates: dict[str, tuple[str, float]] = {base_source: (text, round(confidence, 4))}
    for candidate in candidates_list:
        if candidate.item is None:
            continue
        suffix = "crop" if candidate.variant == "original" else candidate.variant
        candidates[f"{base_source}_{suffix}"] = (
            candidate.item.text,
            round(candidate.item.confidence, 4),
        )

    source = base_source
    best = pick_best(candidates_list)
    if best is not None and best.item.confidence > confidence:
        text, confidence = best.item.text, best.item.confidence
        suffix = "crop" if best.variant == "original" else best.variant
        source = f"{base_source}_{suffix}"
        status = classify_status(confidence, config)

    return TextLine(
        page=page,
        bbox=page_bbox,
        text=text,
        confidence=confidence,
        source=source,
        status=status,
        candidates=candidates,
    )


def collect_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return [path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU 기반 완전 로컬 OCR (Stage 1)")
    parser.add_argument("input", type=Path, help="이미지/PDF 파일 또는 폴더")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--mode",
        choices=["accuracy", "speed"],
        default="accuracy",
        help="Stage 1은 단일 엔진만 사용하므로 현재는 두 모드 동작이 동일하다 (Stage 3부터 분기).",
    )
    parser.add_argument("--lang", default="korean")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"처리할 파일을 찾지 못했습니다: {args.input}", file=sys.stderr)
        return 1

    config = PipelineConfig(mode=args.mode, lang=args.lang, dpi=args.dpi)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("PaddleOCR 엔진 초기화 중 (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)...")
    engine = PaddleOCREngine(config)

    for input_path in inputs:
        start = time.monotonic()
        print(f"[처리 중] {input_path}")
        doc = run_pipeline(input_path, config, engine=engine)

        stem = input_path.stem
        txt_path = args.output_dir / f"{stem}.txt"
        json_path = args.output_dir / f"{stem}.json"
        save_txt(txt_path, doc.final_text)
        save_json(json_path, doc)

        elapsed = time.monotonic() - start
        print(f"  -> {txt_path}")
        print(f"  -> {json_path}")
        print(f"  ({elapsed:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
