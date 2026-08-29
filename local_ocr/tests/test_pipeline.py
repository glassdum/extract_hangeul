"""FakeOCREngine으로 실제 PaddleOCR 모델 없이 파이프라인 배선(입력->인식->정렬->결과)만 검증한다.

실제 PaddleOCR 인식 품질 검증은 모델 가중치를 내려받을 수 있는 환경에서
`python app.py <파일>`로 직접 확인해야 한다 (README 참고).
"""

import pymupdf
from PIL import Image

from app import run_pipeline
from common.config import PipelineConfig
from recognition.base import OCREngine, RecognizedItem


class FakeOCREngine(OCREngine):
    def __init__(self, items):
        self._items = items

    def recognize(self, image):
        return self._items


def test_run_pipeline_on_image_classifies_by_confidence(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    items = [
        RecognizedItem(text="hello", confidence=0.95, polygon=[[10, 10], [60, 10], [60, 30], [10, 30]]),
        RecognizedItem(text="world", confidence=0.4, polygon=[[10, 50], [60, 50], [60, 70], [10, 70]]),
    ]
    config = PipelineConfig()

    doc = run_pipeline(img_path, config, engine=FakeOCREngine(items))

    assert doc.final_text == "hello world"
    lines = doc.pages[0].lines
    assert lines[0].status == "auto_confirmed"
    assert lines[1].status == "low_confidence"


def test_run_pipeline_on_mixed_pdf_combines_text_layer_and_ocr(tmp_path):
    img_src = tmp_path / "embedded.png"
    Image.new("RGB", (100, 40), "white").save(img_src)

    pdf_path = tmp_path / "mixed.pdf"
    doc_pdf = pymupdf.open()
    page = doc_pdf.new_page(width=400, height=300)
    page.insert_text((30, 40), "direct text", fontsize=18, fontname="helv")
    page.insert_image(pymupdf.Rect(30, 100, 130, 140), filename=str(img_src))
    doc_pdf.save(pdf_path)
    doc_pdf.close()

    items = [
        RecognizedItem(text="cropped", confidence=0.9, polygon=[[0, 0], [50, 0], [50, 20], [0, 20]])
    ]
    config = PipelineConfig(dpi=72)

    doc = run_pipeline(pdf_path, config, engine=FakeOCREngine(items))

    sources = {line.source for line in doc.all_lines()}
    assert sources == {"pdf_text", "paddle_print"}
    assert "direct text" in doc.final_text
    assert "cropped" in doc.final_text
