import numpy as np

from recognition.base import OCREngine, RecognizedItem
from recognition.caching_engine import CachingEngine
from storage.cache import ResultCache


class CountingEngine(OCREngine):
    """호출될 때마다 confidence를 하나씩 올려서, 캐시 히트/미스를 결과로 구분할 수 있게 한다."""

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images):
        results = []
        for _ in images:
            self.calls += 1
            results.append([RecognizedItem(text="x", confidence=self.calls, polygon=[])])
        return results


def _img(fill: int) -> np.ndarray:
    return np.full((10, 10, 3), fill, dtype=np.uint8)


def test_second_call_with_same_image_is_served_from_cache(tmp_path):
    inner = CountingEngine()
    engine = CachingEngine(inner, ResultCache(tmp_path / "cache.sqlite3"))

    first = engine.recognize(_img(1))
    second = engine.recognize(_img(1))

    assert inner.calls == 1
    assert first[0].confidence == second[0].confidence == 1


def test_different_images_both_hit_the_underlying_engine(tmp_path):
    inner = CountingEngine()
    engine = CachingEngine(inner, ResultCache(tmp_path / "cache.sqlite3"))

    engine.recognize(_img(1))
    engine.recognize(_img(2))

    assert inner.calls == 2


def test_recognize_batch_only_calls_engine_for_uncached_images(tmp_path):
    inner = CountingEngine()
    cache = ResultCache(tmp_path / "cache.sqlite3")
    engine = CachingEngine(inner, cache)

    engine.recognize(_img(1))  # 미리 캐시에 채워 둔다
    results = engine.recognize_batch([_img(1), _img(2), _img(1)])

    assert inner.calls == 2  # img(2)만 새로 계산됨
    assert results[0][0].confidence == results[2][0].confidence  # 캐시된 동일 결과


def test_cache_is_shared_across_engine_instances_via_same_db(tmp_path):
    db_path = tmp_path / "cache.sqlite3"
    inner1 = CountingEngine()
    CachingEngine(inner1, ResultCache(db_path)).recognize(_img(1))

    inner2 = CountingEngine()
    result = CachingEngine(inner2, ResultCache(db_path)).recognize(_img(1))

    assert inner2.calls == 0  # 새 엔진 인스턴스이지만 캐시를 그대로 재사용
    assert result[0].confidence == 1


def test_different_namespaces_do_not_share_cache_entries(tmp_path):
    cache = ResultCache(tmp_path / "cache.sqlite3")
    inner_a = CountingEngine()
    inner_b = CountingEngine()

    CachingEngine(inner_a, cache, namespace="engine_a").recognize(_img(1))
    CachingEngine(inner_b, cache, namespace="engine_b").recognize(_img(1))

    assert inner_a.calls == 1
    assert inner_b.calls == 1  # namespace가 다르므로 캐시를 공유하지 않는다
