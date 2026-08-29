"""이미지 전처리.

문서 "이미지 전처리" 표 중 Stage 1이 다루는 범위는 정규화(RGB 변환·EXIF
회전·Alpha 합성, `input.image_loader`에서 이미 처리됨)와 문서/줄 방향 보정
(PaddleOCR 파이프라인의 `use_doc_orientation_classify`/`use_textline_orientation`
로 처리, `recognition.paddle_engine` 참고)뿐이다.

Deskew, Perspective Transform, CLAHE/Gamma, Denoising, Otsu/Adaptive
Threshold, 확대, 선 제거, 반전은 문서의 Stage 2("전처리 탐색")에서
Crop 단위 다중 보정본 비교와 함께 구현될 예정이며 아직 없다.
"""

from .convert import to_bgr_ndarray

__all__ = ["to_bgr_ndarray"]
