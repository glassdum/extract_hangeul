"""python -m review <json_path> [--history-db path] 로 검토 화면을 연다."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .main_window import ReviewWindow


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR 결과 검토 화면 (Stage 6)")
    parser.add_argument("json_path", help="app.py가 만든 JSON 결과 파일")
    parser.add_argument(
        "--history-db", default=None, help="수정 이력을 남길 SQLite 파일 경로 (생략 가능)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    app = QApplication(sys.argv[:1])
    window = ReviewWindow(args.json_path, history_db_path=args.history_db)
    window.resize(1000, 650)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
