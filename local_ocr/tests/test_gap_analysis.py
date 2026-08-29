from common.types import BBox, TextLine
from spacing.gap_analysis import estimate_char_width, should_glue


def _line(x0, x1, text, y0=0, y1=20):
    return TextLine(page=1, bbox=BBox(x0, y0, x1, y1), text=text, confidence=0.9, source="test")


def test_estimate_char_width_divides_box_width_by_length():
    line = _line(0, 40, "abcd")  # width=40, 4 chars -> 10/char
    assert estimate_char_width(line) == 10.0


def test_estimate_char_width_zero_for_empty_text():
    line = _line(0, 40, "")
    assert estimate_char_width(line) == 0.0


def test_should_glue_true_when_gap_touches_or_overlaps():
    left = _line(0, 40, "abcd")  # x1 = 40
    right = _line(40, 80, "efgh")  # x0 = 40, gap == 0
    assert should_glue(left, right) is True

    overlapping = _line(35, 80, "efgh")  # gap negative
    assert should_glue(left, overlapping) is True


def test_should_glue_true_when_gap_much_narrower_than_char_width():
    # char width ~10, gap=1 (10% of char width) -> 검출기가 한 단어를 쪼갠 것으로 본다
    left = _line(0, 40, "abcd")
    right = _line(41, 81, "efgh")
    assert should_glue(left, right) is True


def test_should_glue_false_for_typical_word_gap():
    # char width ~10, gap=8 (80% of char width) -> 일반적인 단어 간격
    left = _line(0, 40, "abcd")
    right = _line(48, 88, "efgh")
    assert should_glue(left, right) is False


def test_should_glue_false_when_no_width_evidence_available():
    left = _line(0, 40, "")  # 빈 텍스트 -> 글자 폭 추정 불가
    right = _line(45, 85, "")
    assert should_glue(left, right) is False
