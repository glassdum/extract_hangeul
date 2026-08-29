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


def test_join_text_glues_adjacent_boxes_with_tiny_gap_same_row():
    # 글자 폭 ~10(width40/len4). 간격 1은 검출기가 "abcdefgh"를 둘로 쪼갠 경우다.
    lines = [
        _line(1, 0, 0, 40, 20, "abcd"),
        _line(1, 41, 0, 81, 20, "efgh"),
    ]
    assert join_text(lines) == "abcdefgh"


def test_join_text_keeps_space_for_normal_word_gap_same_row():
    lines = [
        _line(1, 0, 0, 40, 20, "abcd"),
        _line(1, 48, 0, 88, 20, "efgh"),
    ]
    assert join_text(lines) == "abcd efgh"


def test_join_text_always_spaces_across_different_rows_even_with_tiny_x_gap():
    # x축 간격만 보면 붙어야 할 것 같아도, 행이 다르면(줄바꿈) 항상 공백 하나다.
    lines = [
        _line(1, 0, 0, 40, 20, "abcd"),
        _line(1, 41, 100, 81, 120, "efgh"),
    ]
    assert join_text(lines) == "abcd efgh"
