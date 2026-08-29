"""PDF·이미지 로딩 (문서 "처리 파이프라인" 1~2단계, "파일 유형별 입력 처리")."""

from .loader import detect_input_type, load_document

__all__ = ["detect_input_type", "load_document"]
