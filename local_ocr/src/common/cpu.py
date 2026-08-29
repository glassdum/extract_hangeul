"""CPU 스레드 수 결정 (문서 "CPU 성능 최적화": "실제 CPU 코어 수에 따라 Thread를 제한한다").

설치된 paddleocr==3.7.0을 직접 확인해 보면 `cpu_threads`의 라이브러리 기본값은
실제 코어 수와 무관하게 고정된 10이다 (`paddleocr/_constants.py:
DEFAULT_CPU_THREADS = 10`) — 코어가 10개보다 적은 기기에서는 스레드가
코어 수보다 많아져 컨텍스트 스위칭 비용만 늘고, 많은 기기에서는 여유
코어를 못 쓴다. 그래서 실제 코어 수를 물어 직접 넘긴다.
"""

from __future__ import annotations

import os


def recommended_thread_count(reserve: int = 1) -> int:
    """`os.cpu_count() - reserve`(최소 1). 코어 하나는 GUI/OS 몫으로 남겨 둔다."""
    total = os.cpu_count() or 1
    return max(1, total - reserve)
