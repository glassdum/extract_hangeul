from ensemble.cross_check import cross_check_texts, merge_with_markers


def test_merge_with_markers_identical_strings():
    merged, had_mismatch = merge_with_markers("홍길동", "홍길동")
    assert merged == "홍길동"
    assert had_mismatch is False


def test_merge_with_markers_single_char_difference_matches_doc_example():
    merged, had_mismatch = merge_with_markers("홍길동", "홍김동")
    assert merged == "홍[판독 불가]동"
    assert had_mismatch is True


def test_merge_with_markers_collapses_consecutive_mismatches_into_one_marker():
    merged, had_mismatch = merge_with_markers("010-1234-5678", "010-19x4-5678")
    assert merged == "010-1[판독 불가]4-5678"
    assert merged.count("[판독 불가]") == 1
    assert had_mismatch is True


def test_merge_with_markers_handles_length_difference():
    merged, had_mismatch = merge_with_markers("공항대로", "공항로")
    assert merged == "공항[판독 불가]로"
    assert had_mismatch is True


def test_cross_check_texts_exact_match_auto_confirms():
    result = cross_check_texts("홍길동", "홍길동")
    assert result.status == "auto_confirmed"
    assert result.text == "홍길동"


def test_cross_check_texts_partial_mismatch_is_unreadable():
    result = cross_check_texts("홍길동", "홍김동")
    assert result.status == "unreadable"
    assert result.text == "홍[판독 불가]동"


def test_cross_check_texts_wildly_different_strings_go_to_review():
    result = cross_check_texts("홍길동", "완전히 다른 내용의 문장입니다")
    assert result.status == "review_required"
    assert result.text == "홍길동"  # 원본(primary) 값을 그대로 보존한다


def test_cross_check_texts_normalizes_whitespace_and_nfc():
    result = cross_check_texts("  홍길동 ", "홍길동")
    assert result.status == "auto_confirmed"
    assert result.text == "홍길동"
