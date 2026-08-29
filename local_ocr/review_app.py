#!/usr/bin/env python3
"""검토 화면(Stage 6) 진입점. `src/review/__main__.py`를 감싼 얇은 wrapper다 —
app.py와 마찬가지로 `python -m review ...`처럼 `src`를 PYTHONPATH에 직접
넣지 않아도 이 파일 하나로 바로 실행할 수 있게 한다.

Usage:
    python review_app.py output/input.json [--history-db output/history.sqlite3]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from review.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
