"""OCR 엔진 결과 캐시 래퍼 (문서 "CPU 성능 최적화": Hash 기반 재계산 방지).

`OCREngine`을 그대로 감싸므로 `recognize()`/`recognize_batch()` 어느 쪽으로
불려도 캐시가 적용된다 — `run_pipeline`의 1차 페이지 인식과 Stage 2/3의
Crop 재판독 모두 결국 이 두 메서드로 수렴하기 때문에, 페이지 단위든 Crop
단위든 따로 처리할 필요가 없다.
"""

from __future__ import annotations

import numpy as np

from storage.cache import ResultCache, hash_image

from .base import OCREngine, RecognizedItem


class CachingEngine(OCREngine):
    def __init__(self, engine: OCREngine, cache: ResultCache, namespace: str | None = None):
        self._engine = engine
        self._cache = cache
        self._namespace = namespace or type(engine).__name__

    def recognize(self, image: np.ndarray) -> list[RecognizedItem]:
        return self.recognize_batch([image])[0]

    def recognize_batch(self, images: list[np.ndarray]) -> list[list[RecognizedItem]]:
        keys = [hash_image(image, self._namespace) for image in images]
        cached = [self._cache.get(key) for key in keys]
        results: list[list[RecognizedItem] | None] = [
            _from_cache(value) if value is not None else None for value in cached
        ]

        missing_indices = [i for i, r in enumerate(results) if r is None]
        if missing_indices:
            fresh = self._engine.recognize_batch([images[i] for i in missing_indices])
            for i, items in zip(missing_indices, fresh):
                results[i] = items
                self._cache.set(keys[i], _to_cache(items))

        return results  # type: ignore[return-value]


def _to_cache(items: list[RecognizedItem]) -> list[dict]:
    return [{"text": item.text, "confidence": item.confidence, "polygon": item.polygon} for item in items]


def _from_cache(value: list[dict]) -> list[RecognizedItem]:
    return [RecognizedItem(**item) for item in value]
