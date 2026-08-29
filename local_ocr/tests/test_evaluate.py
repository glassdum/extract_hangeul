from evaluate import edit_distance, evaluate, load_label_file


def test_edit_distance_identical_strings():
    assert edit_distance("홍길동", "홍길동") == 0


def test_edit_distance_single_substitution():
    assert edit_distance("홍길동", "홍김동") == 1


def test_edit_distance_insertion_and_deletion():
    assert edit_distance("abc", "ab") == 1
    assert edit_distance("ab", "abc") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3


def test_load_label_file_normalizes_whitespace_and_nfc(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("a.png\t 홍길동 \nb.png\t world\n", encoding="utf-8")

    labels = load_label_file(path)

    assert labels == {"a.png": "홍길동", "b.png": "world"}


def test_evaluate_computes_cer_and_exact_match():
    references = {"a.png": "홍길동", "b.png": "world"}
    hypotheses = {"a.png": "홍길동", "b.png": "wor1d"}

    result = evaluate(references, hypotheses)

    assert result["num_samples"] == 2
    assert result["num_missing_predictions"] == 0
    assert result["exact_match"] == 0.5
    # 총 편집 거리 1(world->wor1d) / 총 정답 글자 수 8(홍길동=3 + world=5)
    assert result["cer"] == 1 / 8


def test_evaluate_reports_missing_predictions_without_crashing():
    references = {"a.png": "x", "b.png": "y"}
    hypotheses = {"a.png": "x"}

    result = evaluate(references, hypotheses)

    assert result["num_samples"] == 1
    assert result["num_missing_predictions"] == 1
    assert result["exact_match"] == 1.0
