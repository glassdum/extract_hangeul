import pymupdf
from PIL import Image

from input.pdf_loader import load_pdf


def test_text_pdf_extracts_words_directly_without_ocr_regions(tmp_path):
    pdf_path = tmp_path / "text.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((30, 40), "Hello World", fontsize=18, fontname="helv")
    doc.save(pdf_path)
    doc.close()

    pages = load_pdf(pdf_path, dpi=300, min_text_layer_chars=10)

    assert len(pages) == 1
    assert pages[0].ocr_regions == []
    texts = [line.text for line in pages[0].text_lines]
    assert texts == ["Hello World"]
    assert pages[0].text_lines[0].source == "pdf_text"
    assert pages[0].text_lines[0].confidence == 1.0


def test_scan_pdf_without_text_layer_becomes_single_ocr_region(tmp_path):
    img_path = tmp_path / "scan_src.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    pdf_path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.insert_image(pymupdf.Rect(0, 0, 200, 100), filename=str(img_path))
    doc.save(pdf_path)
    doc.close()

    pages = load_pdf(pdf_path, dpi=300, min_text_layer_chars=10)

    assert len(pages) == 1
    assert pages[0].text_lines == []
    assert len(pages[0].ocr_regions) == 1
    region = pages[0].ocr_regions[0]
    assert region.image.size == (pages[0].width, pages[0].height)


def test_mixed_pdf_extracts_text_and_embedded_image_region(tmp_path):
    img_path = tmp_path / "embedded.png"
    Image.new("RGB", (100, 50), "white").save(img_path)

    pdf_path = tmp_path / "mixed.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    page.insert_text((30, 40), "top text", fontsize=18, fontname="helv")
    page.insert_image(pymupdf.Rect(30, 80, 130, 130), filename=str(img_path))
    doc.save(pdf_path)
    doc.close()

    pages = load_pdf(pdf_path, dpi=72, min_text_layer_chars=5)  # 72dpi -> zoom=1 so px == pt

    assert len(pages) == 1
    page_result = pages[0]
    assert [line.text for line in page_result.text_lines] == ["top text"]
    assert len(page_result.ocr_regions) == 1
    region = page_result.ocr_regions[0]
    assert region.bbox.x0 == 30
    assert region.bbox.y0 == 80
    assert region.bbox.x1 == 130
    assert region.bbox.y1 == 130
