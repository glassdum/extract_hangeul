"""결과·후보·수정 이력 저장 (TXT·JSON). SQLite 이력 저장은 Stage 6(검토 GUI)에서 추가된다."""

from .writer import save_json, save_txt

__all__ = ["save_txt", "save_json"]
