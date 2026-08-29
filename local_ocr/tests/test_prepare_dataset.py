"""resources/handwriting_training/prepare_dataset.py 검증.

실제 AI Hub 데이터는 없지만, canonical manifest -> writer 단위 분할 ->
PaddleOCR 라벨 형식 변환 로직 자체는 합성 데이터로 완전히 검증할 수 있다.
"""

import json

from prepare_dataset import (
    Sample,
    copy_images,
    load_manifest,
    split_by_writer,
    write_dict_file,
    write_label_file,
)


def _write_manifest(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_load_manifest_parses_records(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"image_path": "a.png", "text": "가나", "writer_id": "w1"},
            {"image_path": "b.png", "text": "다라"},  # writer_id 생략 가능
        ],
    )

    samples = load_manifest(manifest)

    assert samples == [
        Sample(image_path="a.png", text="가나", writer_id="w1"),
        Sample(image_path="b.png", text="다라", writer_id=""),
    ]


def test_load_manifest_raises_on_missing_required_field(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image_path": "a.png"}\n', encoding="utf-8")

    try:
        load_manifest(manifest)
        assert False, "should have raised"
    except ValueError as exc:
        assert "text" in str(exc)


def test_split_by_writer_keeps_same_writer_in_one_split_only():
    samples = [
        Sample(image_path=f"w1_{i}.png", text="x", writer_id="w1") for i in range(4)
    ] + [Sample(image_path=f"w2_{i}.png", text="x", writer_id="w2") for i in range(4)]

    train, val, test = split_by_writer(samples, val_ratio=0.5, test_ratio=0.0, seed=1)

    train_writers = {s.writer_id for s in train}
    val_writers = {s.writer_id for s in val}
    test_writers = {s.writer_id for s in test}

    assert not (train_writers & val_writers)
    assert not (train_writers & test_writers)
    assert not (val_writers & test_writers)
    assert len(train) + len(val) + len(test) == len(samples)


def test_split_by_writer_puts_writerless_samples_always_in_train():
    samples = [Sample(image_path="print.png", text="인쇄체", writer_id="")]
    train, val, test = split_by_writer(samples, val_ratio=0.5, test_ratio=0.5, seed=1)

    assert train == samples
    assert val == []
    assert test == []


def test_copy_images_mirrors_relative_paths(tmp_path):
    images_root = tmp_path / "raw"
    (images_root / "sub").mkdir(parents=True)
    (images_root / "sub" / "a.png").write_bytes(b"fake-image-bytes")

    output_dir = tmp_path / "out"
    samples = [Sample(image_path="sub/a.png", text="x", writer_id="")]

    copy_images(samples, images_root, output_dir)

    copied = output_dir / "sub" / "a.png"
    assert copied.exists()
    assert copied.read_bytes() == b"fake-image-bytes"


def test_write_label_file_uses_tab_separated_format(tmp_path):
    path = tmp_path / "train.txt"
    write_label_file(path, [Sample(image_path="a.png", text="홍길동", writer_id="")])

    assert path.read_text(encoding="utf-8") == "a.png\t홍길동\n"


def test_write_dict_file_copies_provided_dict(tmp_path):
    dict_src = tmp_path / "official_dict.txt"
    dict_src.write_text("가\n나\n", encoding="utf-8")
    out = tmp_path / "dict.txt"

    write_dict_file(out, samples=[], dict_path=dict_src)

    assert out.read_text(encoding="utf-8") == "가\n나\n"


def test_write_dict_file_falls_back_to_generating_from_samples(tmp_path, capsys):
    out = tmp_path / "dict.txt"
    samples = [Sample(image_path="a.png", text="ba", writer_id="")]

    write_dict_file(out, samples=samples, dict_path=None)

    assert out.read_text(encoding="utf-8") == "a\nb\n"
    assert "경고" in capsys.readouterr().out
