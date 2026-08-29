from storage.history import CorrectionRecord, HistoryStore


def test_add_and_retrieve_correction(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = CorrectionRecord(
        source_path="sample.png",
        page=1,
        bbox=(1.0, 2.0, 3.0, 4.0),
        before_text="홍깈동",
        before_status="low_confidence",
        after_text="홍길동",
        after_status="auto_confirmed",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    store.add(record)
    rows = store.all()

    assert len(rows) == 1
    assert rows[0]["source_path"] == "sample.png"
    assert rows[0]["before_text"] == "홍깈동"
    assert rows[0]["after_text"] == "홍길동"
    assert rows[0]["bbox_x0"] == 1.0
    assert rows[0]["bbox_y1"] == 4.0


def test_store_persists_across_instances(tmp_path):
    db_path = tmp_path / "sub" / "history.sqlite3"
    store1 = HistoryStore(db_path)
    store1.add(
        CorrectionRecord(
            source_path="a.png",
            page=1,
            bbox=(0, 0, 1, 1),
            before_text="a",
            before_status="review_required",
            after_text="b",
            after_status="auto_confirmed",
        )
    )

    store2 = HistoryStore(db_path)  # 같은 파일을 다시 연다
    assert len(store2.all()) == 1


def test_add_generates_timestamp_when_not_given(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.add(
        CorrectionRecord(
            source_path="a.png",
            page=1,
            bbox=(0, 0, 1, 1),
            before_text="a",
            before_status="review_required",
            after_text="b",
            after_status="auto_confirmed",
        )
    )
    assert store.all()[0]["timestamp"]  # 비어 있지 않다
