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
    """첫 호출(1차 전체 영역 인식)은 `first_call_items`를, 이후 호출(Stage 2 재판독의
    원본/보정본 Crop마다)은 `reprocess_item`을 반환한다."""

    def __init__(self, first_call_items, reprocess_item=None):
        self._first_call_items = first_call_items
        self._reprocess_item = reprocess_item
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        if self.calls == 1:
            return self._first_call_items
        return [self._reprocess_item] if self._reprocess_item is not None else []


def test_run_pipeline_auto_confirms_high_confidence_without_reprocessing(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    items = [
        RecognizedItem(text="hello", confidence=0.95, polygon=[[10, 10], [60, 10], [60, 30], [10, 30]]),
    ]
    engine = FakeOCREngine(items)

    doc = run_pipeline(img_path, PipelineConfig(), engine=engine)

    assert doc.final_text == "hello"
    line = doc.pages[0].lines[0]
    assert line.status == "auto_confirmed"
    assert line.candidates == {}  # 재판독을 거치지 않았다.
    assert engine.calls == 1  # Stage 2 재판독 호출이 없었다.


def test_run_pipeline_reprocesses_low_confidence_and_upgrades_status(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    items = [
        RecognizedItem(text="world", confidence=0.4, polygon=[[10, 50], [60, 50], [60, 70], [10, 70]]),
    ]
    reprocess_item = RecognizedItem(
        text="world", confidence=0.85, polygon=[[0, 0], [50, 0], [50, 20], [0, 20]]
    )
    engine = FakeOCREngine(items, reprocess_item=reprocess_item)

    doc = run_pipeline(img_path, PipelineConfig(), engine=engine)

    line = doc.pages[0].lines[0]
    assert line.text == "world"
    assert line.confidence == 0.85
    assert line.status == "auto_confirmed"
    assert engine.calls > 1  # 원본 + 전처리 보정본들에 대해 재판독했다.
    assert len(line.candidates) > 1  # 여러 보정본 후보가 JSON용으로 남았다.
    assert line.candidates["paddle_print"] == ("world", 0.4)  # 1차 인식값도 후보로 남는다.
    assert "paddle_print_crop" in line.candidates  # 보정 없는 재크롭 시도도 남는다.


class FixedTesseractStub(OCREngine):
    """호출할 때마다 항상 같은 결과를 돌려주는 단순 스텁 (Tesseract 대역)."""

    def __init__(self, item):
        self._item = item

    def recognize(self, image):
        return [self._item] if self._item is not None else []


def test_run_pipeline_cross_check_auto_confirms_when_engines_agree(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    paddle_item = RecognizedItem(
        text="hello", confidence=0.4, polygon=[[10, 10], [60, 10], [60, 30], [10, 30]]
    )
    paddle_engine = FakeOCREngine([paddle_item], reprocess_item=paddle_item)
    tesseract_engine = FixedTesseractStub(
        RecognizedItem(text="hello", confidence=0.9, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])
    )

    doc = run_pipeline(
        img_path, PipelineConfig(), engine=paddle_engine, tesseract_engine=tesseract_engine
    )

    line = doc.pages[0].lines[0]
    assert line.status == "auto_confirmed"
    assert line.text == "hello"
    assert line.candidates["tesseract"] == ("hello", 0.9)


def test_run_pipeline_cross_check_marks_unreadable_on_partial_mismatch(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    paddle_item = RecognizedItem(
        text="hollo", confidence=0.4, polygon=[[10, 10], [60, 10], [60, 30], [10, 30]]
    )
    paddle_engine = FakeOCREngine([paddle_item], reprocess_item=paddle_item)
    tesseract_engine = FixedTesseractStub(
        RecognizedItem(text="hello", confidence=0.9, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]])
    )

    doc = run_pipeline(
        img_path, PipelineConfig(), engine=paddle_engine, tesseract_engine=tesseract_engine
    )

    line = doc.pages[0].lines[0]
    assert line.status == "unreadable"
    assert "[판독 불가]" in line.text


def test_run_pipeline_cross_check_review_required_on_disagreement(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    paddle_item = RecognizedItem(
        text="hello", confidence=0.4, polygon=[[10, 10], [60, 10], [60, 30], [10, 30]]
    )
    paddle_engine = FakeOCREngine([paddle_item], reprocess_item=paddle_item)
    tesseract_engine = FixedTesseractStub(
        RecognizedItem(
            text="completely different sentence",
            confidence=0.9,
            polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
        )
    )

    doc = run_pipeline(
        img_path, PipelineConfig(), engine=paddle_engine, tesseract_engine=tesseract_engine
    )

    line = doc.pages[0].lines[0]
    assert line.status == "review_required"
    assert line.text == "hello"  # 최선의 추정값(Paddle)을 그대로 보존한다


def test_run_pipeline_without_tesseract_engine_skips_stage3(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    paddle_item = RecognizedItem(
        text="hello", confidence=0.4, polygon=[[10, 10], [60, 10], [60, 30], [10, 30]]
    )
    paddle_engine = FakeOCREngine([paddle_item], reprocess_item=paddle_item)

    doc = run_pipeline(img_path, PipelineConfig(), engine=paddle_engine, tesseract_engine=None)

    line = doc.pages[0].lines[0]
    assert "tesseract" not in line.candidates


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
