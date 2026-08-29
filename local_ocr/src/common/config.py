"""Pipeline-wide configuration.

Values mirror the "확정 요구사항" / "CPU 성능 최적화" sections of the
technical development plan. Only Stage 1 (입력·기본 OCR) knobs are wired
up today; later stages (전처리 프로파일, ensemble 임계값, 손글씨 모델 경로 등)
will extend this dataclass rather than replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .cpu import recommended_thread_count

Mode = Literal["accuracy", "speed"]
InferenceBackend = Literal["paddle", "onnxruntime"]

# 문서 "신뢰도와 판독 불가 정책" 표를 Stage 1에서 쓸 수 있는 형태로 근사한 값.
# 실제 앙상블(모델 간 교차 검증) 기반 판정은 Stage 3에서 대체된다.
DEFAULT_AUTO_CONFIRM_THRESHOLD = 0.80
DEFAULT_REVIEW_THRESHOLD = 0.50

# 문서 "파일 유형별 입력 처리" 표.
DEFAULT_DPI = 300
SMALL_TEXT_DPI = 450

# 문서 "확정 요구사항": 텍스트 PDF 판정 시 사용하는 최소 글자 수 기준.
MIN_TEXT_LAYER_CHARS = 10

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".webp", ".heic", ".heif", ".gif",
    ".psd",  # Pillow의 PsdImagePlugin으로 합성 미리보기만 읽음 (숨겨진 레이어 제외)
}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}


@dataclass
class PipelineConfig:
    """Stage 1 파이프라인 실행 설정."""

    # 문서 "기본 설정": 정확도 우선 모드가 기본값.
    mode: Mode = "accuracy"

    # PaddleOCR 인식 언어 팩. "korean" 팩이 한글·영문·숫자를 함께 지원한다.
    lang: str = "korean"

    dpi: int = DEFAULT_DPI
    small_text_dpi: int = SMALL_TEXT_DPI
    min_text_layer_chars: int = MIN_TEXT_LAYER_CHARS

    auto_confirm_threshold: float = DEFAULT_AUTO_CONFIRM_THRESHOLD
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD

    # Stage 3: Paddle 결과가 불확실할 때 Tesseract로 교차 판독할지 여부.
    # 문서 "모델 선택 근거": Tesseract는 보조 엔진이므로 끌 수 있게 둔다
    # (Tesseract 미설치 환경에서 개발/테스트할 때 유용).
    enable_cross_check: bool = True
    tesseract_lang: str = "kor+eng"

    # 문서 "CPU 성능 최적화": GPU 불필요, 실제 코어 수 기반 스레드 제한.
    # 설치된 paddleocr==3.7.0의 라이브러리 기본값(DEFAULT_CPU_THREADS=10,
    # paddleocr/_constants.py에서 직접 확인)은 실제 코어 수와 무관한 고정값이라
    # 대신 계산한 값을 기본으로 쓴다.
    use_gpu: bool = False
    cpu_threads: int = field(default_factory=recommended_thread_count)

    # PaddleOCR 파이프라인 옵션 (문서 "사용 모델 구성" 표의 문서 방향/줄 방향/문서 펴기 단계).
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = False  # 조건부 실행: 기본은 끔, 왜곡 감지 시 stage2에서 활성화 예정.
    use_textline_orientation: bool = True

    # 문서 "CPU 성능 최적화": "ONNX Runtime의 CPUExecutionProvider를 사용".
    # 설치된 paddleocr._common_args.SUPPORTED_INFERENCE_ENGINE_LIST에 실제로
    # "onnxruntime"이 있음을 확인하고 그대로 연결했다 — 다만 onnx 형식으로
    # 변환된 모델과 onnxruntime 패키지가 있어야 동작하며, 이 저장소를 만든
    # 샌드박스에서는 그 모델을 받을 수 없어 실제 추론까지는 검증하지 못했다.
    inference_backend: InferenceBackend = "paddle"

    # 문서 "CPU 성능 최적화": "처리된 페이지와 Crop의 Hash를 저장해 동일 입력을
    # 다시 계산하지 않는다."
    enable_cache: bool = True
    cache_path: Path | None = None  # None이면 호출부(app.py)가 output-dir 기준으로 정한다.

    resources_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "resources"
    )

    def __post_init__(self) -> None:
        if self.mode not in ("accuracy", "speed"):
            raise ValueError(f"unknown mode: {self.mode!r}")
