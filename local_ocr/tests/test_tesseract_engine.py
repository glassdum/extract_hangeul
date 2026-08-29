import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from preprocess.convert import to_bgr_ndarray
from recognition.tesseract_engine import TesseractEngine

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
KOREAN_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

pytestmark = pytest.mark.skipif(
    not TESSERACT_AVAILABLE, reason="시스템에 tesseract-ocr 바이너리가 설치돼 있지 않음"
)


def _render_text(text: str) -> Image.Image:
    font = ImageFont.truetype(KOREAN_FONT_PATH, 40)
    img = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(img).text((10, 20), text, font=font, fill="black")
    return img


def test_tesseract_engine_reads_clear_korean_text():
    engine = TesseractEngine()
    items = engine.recognize(to_bgr_ndarray(_render_text("홍길동")))

    assert len(items) == 1
    assert items[0].text == "홍길동"
    assert items[0].confidence > 0.5


def test_tesseract_engine_returns_empty_list_for_blank_image():
    engine = TesseractEngine()
    blank = Image.new("RGB", (100, 40), "white")
    assert engine.recognize(to_bgr_ndarray(blank)) == []


def test_tesseract_engine_confidence_is_normalized_to_0_1_range():
    engine = TesseractEngine()
    items = engine.recognize(to_bgr_ndarray(_render_text("Hello")))
    assert items
    assert 0.0 <= items[0].confidence <= 1.0
