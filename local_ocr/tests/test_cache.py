import numpy as np

from storage.cache import ResultCache, hash_image


def test_get_returns_none_for_missing_key(tmp_path):
    cache = ResultCache(tmp_path / "cache.sqlite3")
    assert cache.get("missing") is None


def test_set_then_get_round_trips_json_value(tmp_path):
    cache = ResultCache(tmp_path / "cache.sqlite3")
    value = [{"text": "홍길동", "confidence": 0.9, "polygon": [[0, 0], [1, 1]]}]

    cache.set("key1", value)

    assert cache.get("key1") == value


def test_set_overwrites_existing_key(tmp_path):
    cache = ResultCache(tmp_path / "cache.sqlite3")
    cache.set("key1", [{"a": 1}])
    cache.set("key1", [{"a": 2}])
    assert cache.get("key1") == [{"a": 2}]


def test_cache_persists_across_instances(tmp_path):
    db_path = tmp_path / "sub" / "cache.sqlite3"
    ResultCache(db_path).set("key1", ["v"])
    assert ResultCache(db_path).get("key1") == ["v"]


def test_hash_image_is_deterministic_for_same_content():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert hash_image(img, "ns") == hash_image(img.copy(), "ns")


def test_hash_image_differs_by_namespace():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert hash_image(img, "paddle") != hash_image(img, "tesseract")


def test_hash_image_differs_by_pixel_content():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.ones((10, 10, 3), dtype=np.uint8)
    assert hash_image(a, "ns") != hash_image(b, "ns")


def test_hash_image_differs_by_shape_even_with_same_bytes():
    flat = np.zeros((100, 3), dtype=np.uint8)
    reshaped = flat.reshape(10, 10, 3)
    assert hash_image(flat, "ns") != hash_image(reshaped, "ns")
