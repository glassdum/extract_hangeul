"""결과·후보·수정 이력 저장 (문서 "결과 저장 형식": TXT/JSON/SQLite)."""

from .cache import ResultCache, hash_image
from .history import CorrectionRecord, HistoryStore
from .writer import save_json, save_txt

__all__ = [
    "save_txt",
    "save_json",
    "HistoryStore",
    "CorrectionRecord",
    "ResultCache",
    "hash_image",
]
