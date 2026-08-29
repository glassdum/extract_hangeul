"""이미지 전처리 (문서 "이미지 전처리" 표).

- 정규화(RGB 변환·EXIF 회전·Alpha 합성)는 `input.image_loader`에서 처리된다.
- 문서/줄 방향 보정은 PaddleOCR 파이프라인이 담당한다 (`recognition.paddle_engine`).
- 이 패키지는 그 외 행을 구현한다: 기하 보정(Deskew), 명암(CLAHE/Gamma/Stretch),
  노이즈(Median/Bilateral), 이진화(Otsu/Adaptive), 확대(Lanczos/Cubic),
  선 제거(Morphology), 반전(Black-on-white/White-on-black).
- `variants.generate_variants()`가 위 보정들을 조건부로 조합해 "불확실한
  Crop"에 시도할 후보 이미지 묶음을 만든다 — 실제 재인식·최적 후보 선택은
  `ensemble.reprocess`가 담당한다.

Perspective Transform(문서 펴기, UVDoc)은 아직 구현하지 않았다.
"""

from .convert import to_bgr_ndarray
from .crop import crop_with_padding
from .variants import generate_variants

__all__ = ["to_bgr_ndarray", "crop_with_padding", "generate_variants"]
