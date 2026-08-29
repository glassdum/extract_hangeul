"""후보 비교·신뢰도 판정 (Stage 3).

문서 "신뢰도와 판독 불가 정책"의 자동 확정/재처리/사용자 확인/판독 불가 상태를
여러 엔진(PaddleOCR 인쇄체·손글씨 Fine-tuning·Tesseract) 결과의 교차 비교로
판정하는 로직이 여기 들어간다. Stage 1은 단일 엔진만 사용하므로, 그 근사치로
`storage.writer`가 confidence 임계값만으로 상태를 정한다. 아직 코드가 없다.
"""
