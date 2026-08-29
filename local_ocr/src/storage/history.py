"""수정 이력 저장 (SQLite).

문서 "결과 저장 형식"이 저장 기술로 나열한 "SQLite / JSON / TXT" 중 SQLite가
맡는 부분 — "결과·후보·수정 이력"의 수정 이력. TXT/JSON은 이미 `storage.writer`가
맡고 있다. `review.session.ReviewSession`이 사용자가 검토 화면에서 고친
내용을 저장할 때마다 여기 한 줄씩 남긴다.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    page INTEGER NOT NULL,
    bbox_x0 REAL NOT NULL,
    bbox_y0 REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    before_text TEXT NOT NULL,
    before_status TEXT NOT NULL,
    after_text TEXT NOT NULL,
    after_status TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""


@dataclass
class CorrectionRecord:
    source_path: str
    page: int
    bbox: tuple[float, float, float, float]
    before_text: str
    before_status: str
    after_text: str
    after_status: str
    timestamp: str = ""


class HistoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def add(self, record: CorrectionRecord) -> None:
        timestamp = record.timestamp or datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO corrections
                    (source_path, page, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                     before_text, before_status, after_text, after_status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.source_path,
                    record.page,
                    *record.bbox,
                    record.before_text,
                    record.before_status,
                    record.after_text,
                    record.after_status,
                    timestamp,
                ),
            )
            conn.commit()

    def all(self) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM corrections ORDER BY id").fetchall()
            return [dict(row) for row in rows]
