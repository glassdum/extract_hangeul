"""읽기 순서 정렬 및 줄 연결 (문서 "띄어쓰기 및 줄 연결").

줄 단위 텍스트를 위→아래, 같은 행 안에서는 왼쪽→오른쪽으로 정렬하고,
행이 바뀔 때는 공백 하나로 연결하며, 연속 공백을 정규화하고 Unicode NFC로
통일한다. 각 인식 엔진이 자체적으로 출력한 공백(`use_space_char`, 설치된
PaddleX 소스 확인 결과 기본값이 이미 True)은 그대로 신뢰한다.

Stage 5: 같은 행 안에서 인접한 두 검출 박스는 Bounding Box 간격을 평균
글자 폭과 비교해, 간격이 아주 좁으면(검출기가 한 단어를 둘로 쪼갠 경우)
공백 없이 붙이고, 그렇지 않으면 공백을 넣는다 (`gap_analysis.should_glue`).
문자 단위 좌표까지 내려가는 정밀 비교는 하지 않는다 — 자세한 이유는
`gap_analysis`의 모듈 docstring 참고.
"""

from .gap_analysis import estimate_char_width, should_glue
from .reading_order import join_text, order_reading

__all__ = ["join_text", "order_reading", "estimate_char_width", "should_glue"]
