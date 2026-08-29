"""PaddleOCREngine의 배선(생성 kwargs, 배치 호출)을 실제 paddleocr.PaddleOCR
없이 검증한다 — 이 샌드박스에서는 모델 다운로드가 막혀 있어 실제 추론은
`recognition/tesseract_engine.py`처럼 진짜 바이너리로 검증할 수 없다."""

from unittest.mock import MagicMock, patch

import numpy as np

from common.config import PipelineConfig
from recognition.paddle_engine import PaddleOCREngine


def _fake_result(text: str, score: float) -> dict:
    return {
        "rec_texts": [text],
        "rec_scores": [score],
        "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
    }


def test_constructor_passes_cpu_threads_and_engine_kwargs():
    config = PipelineConfig(cpu_threads=3, inference_backend="onnxruntime")
    fake_ctor = MagicMock(return_value=MagicMock())

    with patch("paddleocr.PaddleOCR", fake_ctor):
        PaddleOCREngine(config)

    _, kwargs = fake_ctor.call_args
    assert kwargs["cpu_threads"] == 3
    assert kwargs["engine"] == "onnxruntime"
    assert kwargs["device"] == "cpu"


def test_recognize_batch_calls_predict_once_for_all_images():
    config = PipelineConfig()
    fake_ocr = MagicMock()
    fake_ocr.predict.return_value = [_fake_result("a", 0.9), _fake_result("b", 0.8)]

    with patch("paddleocr.PaddleOCR", return_value=fake_ocr):
        engine = PaddleOCREngine(config)

    images = [np.zeros((5, 10, 3), dtype=np.uint8), np.ones((5, 10, 3), dtype=np.uint8)]
    results = engine.recognize_batch(images)

    fake_ocr.predict.assert_called_once()
    (predict_arg,), _ = fake_ocr.predict.call_args
    assert predict_arg is images  # 리스트 하나로 한 번만 호출됐다

    assert [r[0].text for r in results] == ["a", "b"]


def test_recognize_delegates_to_recognize_batch():
    config = PipelineConfig()
    fake_ocr = MagicMock()
    fake_ocr.predict.return_value = [_fake_result("hello", 0.95)]

    with patch("paddleocr.PaddleOCR", return_value=fake_ocr):
        engine = PaddleOCREngine(config)

    items = engine.recognize(np.zeros((5, 10, 3), dtype=np.uint8))

    assert len(items) == 1
    assert items[0].text == "hello"
    assert items[0].confidence == 0.95


def test_recognize_batch_returns_empty_list_for_no_images():
    config = PipelineConfig()
    with patch("paddleocr.PaddleOCR", return_value=MagicMock()):
        engine = PaddleOCREngine(config)

    assert engine.recognize_batch([]) == []
