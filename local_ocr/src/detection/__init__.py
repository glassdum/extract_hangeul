"""텍스트 영역 탐지.

Stage 1은 `recognition.paddle_engine`이 감싸는 PaddleOCR 통합 파이프라인의
내장 검출기(PP-OCRv5_mobile_det)를 그대로 사용한다. 이 패키지는 "정밀 검출"
(PP-OCRv5_server_det, 누락 의심 영역 재검출)을 독립적으로 제어해야 하는
Stage 2/3에서 채워질 예정이며 아직 코드가 없다.
"""
