"""PySide6 GUI 스모크 테스트 (오프스크린).

시각적 렌더링 결과 자체를 확인할 방법은 없는 헤드리스 환경이라, 위젯이
정상적으로 만들어지고 버튼/선택 조작이 `ReviewSession`에 올바르게
반영되는지(로직 배선)만 확인한다.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from common.types import BBox, DocumentResult, PageResult, TextLine  # noqa: E402
from review.main_window import ReviewWindow  # noqa: E402
from storage.writer import save_json  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_json(tmp_path):
    from PIL import Image

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


def test_window_loads_only_items_needing_review(qapp, tmp_path):
    window = ReviewWindow(_make_json(tmp_path))
    assert window.list_widget.count() == 1
    assert "wor1d" in window.list_widget.item(0).text()


def test_selecting_item_populates_crop_candidates_and_text(qapp, tmp_path):
    window = ReviewWindow(_make_json(tmp_path))
    window.list_widget.setCurrentRow(0)

    assert not window.image_label.pixmap().isNull()
    assert window.candidates_list.count() == 2
    assert window.text_edit.text() == "wor1d"


def test_double_clicking_candidate_fills_text_edit(qapp, tmp_path):
    window = ReviewWindow(_make_json(tmp_path))
    window.list_widget.setCurrentRow(0)
    tesseract_item = next(
        window.candidates_list.item(i)
        for i in range(window.candidates_list.count())
        if window.candidates_list.item(i).text().startswith("tesseract")
    )

    window._on_candidate_double_clicked(tesseract_item)

    assert window.text_edit.text() == "world"


def test_apply_button_confirms_and_removes_item_from_queue(qapp, tmp_path):
    window = ReviewWindow(_make_json(tmp_path))
    window.list_widget.setCurrentRow(0)
    window.text_edit.setText("world")

    window._on_apply()

    assert window.list_widget.count() == 0
    assert window.session.items_needing_review() == []


def test_unreadable_button_keeps_item_in_queue_for_further_review(qapp, tmp_path):
    window = ReviewWindow(_make_json(tmp_path))
    window.list_widget.setCurrentRow(0)

    window._on_mark_unreadable()

    assert window.list_widget.count() == 1  # 판독 불가도 사람이 다시 볼 수 있게 큐에 남는다
    assert window.session.items_needing_review()[0].status == "unreadable"


def test_save_button_writes_json_and_shows_no_crash(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    json_path = _make_json(tmp_path)
    window = ReviewWindow(json_path)
    window.list_widget.setCurrentRow(0)
    window.text_edit.setText("world")
    window._on_apply()

    window._on_save()

    import json

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["final_text"] == "hello world"
