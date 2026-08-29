from common.types import BBox, TextLine
from spacing.reading_order import join_text, order_reading


def _line(page, x0, y0, x1, y1, text):
    return TextLine(page=page, bbox=BBox(x0, y0, x1, y1), text=text, confidence=0.9, source="test")


def test_order_reading_top_to_bottom_left_to_right():
    lines = [
        _line(1, 300, 10, 400, 40, "d"),
        _line(1, 10, 10, 100, 40, "a"),
        _line(1, 10, 60, 100, 90, "e"),
        _line(1, 150, 12, 250, 42, "b"),
    ]
    ordered = order_reading(lines)
    assert [line.text for line in ordered] == ["a", "b", "d", "e"]


def test_join_text_collapses_whitespace_and_normalizes_nfc():
    lines = [
        _line(1, 10, 10, 100, 40, "  hello   "),
        _line(1, 10, 60, 100, 90, "world"),
    ]
    assert join_text(lines) == "hello world"


def test_join_text_orders_across_pages():
    lines = [
        _line(2, 10, 10, 100, 40, "page2"),
        _line(1, 10, 10, 100, 40, "page1"),
    ]
    assert join_text(lines) == "page1 page2"
