"""범용 이미지 해시 기반 결과 캐시 (문서 "CPU 성능 최적화": "처리된 페이지와
Crop의 Hash를 저장해 동일 입력을 다시 계산하지 않는다").

이 모듈은 무엇을 캐싱하는지 모른다 — 그냥 문자열 키에 JSON 직렬화 가능한
값을 저장/조회할 뿐이다. `recognition.caching_engine.CachingEngine`이
OCR 결과(`RecognizedItem`)를 여기 담을 값으로 변환하는 역할을 맡는다
(이 파일이 `recognition`을 몰라도 되게 하기 위해서다).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS result_cache (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
"""


def hash_image(image: np.ndarray, namespace: str) -> str:
    """이미지 픽셀 내용 + 크기 + namespace(보통 엔진 종류)로 캐시 키를 만든다.

    namespace를 섞는 이유: 같은 이미지라도 엔진(Paddle/Tesseract/손글씨)에
    따라 결과가 다르므로, 캐시 키가 엔진을 구분하지 않으면 서로 다른
    엔진의 결과가 뒤섞인다.
    """
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(image.shape).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


class ResultCache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def get(self, key: str) -> Any | None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT value_json FROM result_cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def set(self, key: str, value: Any) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO result_cache (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            conn.commit()
