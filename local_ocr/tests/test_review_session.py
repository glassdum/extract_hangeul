import json

import pymupdf
from PIL import Image

from common.types import BBox, DocumentResult, PageResult, TextLine
from review.session import ReviewSession
from storage.writer import save_json


def _make_json_for_image(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), "white").save(img_path)

    lines = [
        TextLine(
            page=1, bbox=BBox(10, 10, 60, 30), text="hello", confidence=0.95,
            source="paddle_print", status="auto_confirmed",
        ),
        TextLine(
            page=1, bbox=BBox(10, 50, 60, 70), text="wor1d", confidence=0.4,
            source="paddle_print", status="low_confidence",
            candidates={"paddle_print": ("wor1d", 0.4), "tesseract": ("world", 0.9)},
        ),
    ]
    doc = DocumentResult(
        source_path=str(img_path),
        pages=[PageResult(page=1, width=200, height=100, lines=lines)],
        final_text="hello wor1d",
    )
    json_path = tmp_path / "sample.json"
    save_json(json_path, doc)
    return json_path


def test_items_needing_review_filters_by_status(tmp_path):
    session = ReviewSession(_make_json_for_image(tmp_path))
    items = session.items_needing_review()

    assert len(items) == 1
    assert items[0].text == "wor1d"
    assert items[0].status == "low_confidence"
    assert items[0].candidates["tesseract"] == ("world", 0.9)


def test_all_items_returns_every_line(tmp_path):
    session = ReviewSession(_make_json_for_image(tmp_path))
    assert len(session.all_items()) == 2


def test_apply_correction_updates_item_and_payload(tmp_path):
    session = ReviewSession(_make_json_for_image(tmp_path))
    item = session.items_needing_review()[0]

    session.apply_correction(item, "world", status="auto_confirmed")

    assert item.text == "world"
    assert item.status == "auto_confirmed"
    # payload(원본 JSON dict)도 같이 갱신됐는지 확인
    reloaded = session.items_needing_review()
    assert reloaded == []  # 더 이상 검토 대상이 아니다


def test_mark_unreadable_sets_marker_and_status(tmp_path):
    session = ReviewSession(_make_json_for_image(tmp_path))
    item = session.items_needing_review()[0]

    session.mark_unreadable(item)

    assert item.text == "[판독 불가]"
    assert item.status == "unreadable"


def test_save_writes_updated_json_and_txt(tmp_path):
    json_path = _make_json_for_image(tmp_path)
    session = ReviewSession(json_path)
    item = session.items_needing_review()[0]
    session.apply_correction(item, "world")

    session.save()

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["final_text"] == "hello world"
    txt_path = json_path.with_suffix(".txt")
    assert txt_path.read_text(encoding="utf-8") == "hello world"


def test_save_records_history_when_db_path_given(tmp_path):
    json_path = _make_json_for_image(tmp_path)
    session = ReviewSession(json_path)
    item = session.items_needing_review()[0]
    session.apply_correction(item, "world")

    db_path = tmp_path / "history.sqlite3"
    session.save(history_db_path=db_path)

    from storage.history import HistoryStore

    rows = HistoryStore(db_path).all()
    assert len(rows) == 1
    assert rows[0]["before_text"] == "wor1d"
    assert rows[0]["after_text"] == "world"


def test_get_crop_image_from_image_source(tmp_path):
    session = ReviewSession(_make_json_for_image(tmp_path))
    item = session.items_needing_review()[0]

    crop = session.get_crop_image(item, pad_ratio=0.0)

    # crop_with_padding은 int() 절삭 보정으로 각 축에 +1px을 더한다 (pad_ratio와 무관).
    assert crop.size == (51, 21)


def test_get_crop_image_from_pdf_source(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    doc_pdf = pymupdf.open()
    page = doc_pdf.new_page(width=400, height=300)
    page.insert_text((30, 40), "hello", fontsize=18, fontname="helv")
    doc_pdf.save(pdf_path)
    doc_pdf.close()

    line = TextLine(
        page=1, bbox=BBox(30, 20, 130, 60), text="hello", confidence=0.4,
        source="paddle_print", status="low_confidence",
    )
    result = DocumentResult(
        source_path=str(pdf_path),
        pages=[PageResult(page=1, width=400, height=300, lines=[line])],
        final_text="hello",
    )
    json_path = tmp_path / "doc.json"
    save_json(json_path, result)

    session = ReviewSession(json_path)
    item = session.items_needing_review()[0]
    crop = session.get_crop_image(item, pad_ratio=0.0)

    # bbox width/height=100x40, +1px 절삭 보정 (crop_with_padding 참고)
    assert crop.size == (101, 41)


def test_export_correction_to_manifest_writes_image_and_jsonl(tmp_path):
    session = ReviewSession(_make_json_for_image(tmp_path))
    item = session.items_needing_review()[0]
    session.apply_correction(item, "world")

    images_dir = tmp_path / "manifest_images"
    manifest_path = tmp_path / "manifest.jsonl"
    session.export_correction_to_manifest(item, images_dir, manifest_path)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["text"] == "world"
    assert record["writer_id"] == ""
    assert (images_dir / record["image_path"]).exists()
