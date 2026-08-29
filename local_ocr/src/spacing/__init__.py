"""읽기 순서 정렬 및 줄 연결 (문서 "띄어쓰기 및 줄 연결").

Stage 1 범위: 줄 단위 텍스트를 위→아래, 같은 행 안에서는 왼쪽→오른쪽으로
정렬하고, 줄 사이는 공백 하나로 연결하며, 연속 공백을 정규화하고 Unicode
NFC로 통일한다.

문자 Bounding Box 간격을 실제 단어 간격과 비교해 OCR 공백과 이미지 공백의
일치 여부를 검증하는 것(문서 "띄어쓰기 및 줄 연결" 3~4번째 항목)은 Stage 5
범위이며 아직 구현하지 않았다 — 지금은 각 인식 엔진이 자체적으로 출력한
공백(`use_space_char`)을 그대로 신뢰한다.
"""

from .reading_order import join_text, order_reading

__all__ = ["join_text", "order_reading"]
