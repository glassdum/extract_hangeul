"""PySide6 검토 화면 (문서 "검토 GUI": 원본 Crop, 후보, 수정, 판독 불가 처리).

`session.ReviewSession`이 로딩·수정·저장 로직을 전부 담당하며 PySide6 없이도
동작·테스트된다. `main_window.ReviewWindow`는 그 위에 얹은 얇은 Qt 화면이라,
PySide6가 없는 환경(예: 서버)에서도 `review.session`은 그대로 쓸 수 있도록
여기서 import 실패를 조용히 넘긴다.
"""

from .session import ReviewItem, ReviewSession

try:
    from .main_window import ReviewWindow
except ImportError:  # pragma: no cover - PySide6 미설치 환경
    ReviewWindow = None  # type: ignore[assignment]

__all__ = ["ReviewSession", "ReviewItem", "ReviewWindow"]
