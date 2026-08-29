import json

from common.types import BBox, DocumentResult, PageResult, TextLine
from storage.writer import save_json, save_txt


def _sample_doc():
    line = TextLine(
        page=1,
        bbox=BBox(1, 2, 3, 4),
        text="홍길동",
        confidence=0.91,
        source="paddle_print",
        status="auto_confirmed",
    )
    page = PageResult(page=1, width=100, height=200, lines=[line])
    doc = DocumentResult(source_path="sample.png", pages=[page], final_text="홍길동")
    return doc


def test_save_txt_writes_final_text_only(tmp_path):
    doc = _sample_doc()
    out = tmp_path / "out.txt"
    save_txt(out, doc.final_text)
    assert out.read_text(encoding="utf-8") == "홍길동"


def test_save_json_schema(tmp_path):
    doc = _sample_doc()
    out = tmp_path / "out.json"
    save_json(out, doc)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["source"] == "sample.png"
    assert payload["final_text"] == "홍길동"
    page = payload["pages"][0]
    assert page["page"] == 1
    line = page["lines"][0]
    assert line["text"] == "홍길동"
    assert line["status"] == "auto_confirmed"
    assert line["bbox"] == [1, 2, 3, 4]
    assert line["candidates"]["paddle_print"] == ["홍길동", 0.91]
