"""실제 학습된 손글씨 가중치가 없어도 검증 가능한 부분만 다룬다:
- 모델이 없을 때 명확한 에러로 실패하는지 (호출부가 잡아서 건너뛸 수 있게)
- 있을 때 paddleocr.TextRecognition의 rec_text/rec_score 출력을 올바르게
  파싱하는지 (실제 라이브러리는 `unittest.mock`으로 대신한다).
"""

from unittest.mock import patch

import numpy as np
import pytest

from recognition.handwriting_engine import HandwritingEngine, HandwritingModelNotAvailableError


class _FakeTextRecognition:
    def __init__(self, result, **kwargs):
        self._result = result
        self.init_kwargs = kwargs

    def predict(self, image):
        return self._result


def test_raises_when_no_model_files_present(tmp_path):
    empty_dir = tmp_path / "korean_handwriting_recognition"
    empty_dir.mkdir()

    with pytest.raises(HandwritingModelNotAvailableError):
        HandwritingEngine(empty_dir)


def test_raises_when_directory_does_not_exist(tmp_path):
    with pytest.raises(HandwritingModelNotAvailableError):
        HandwritingEngine(tmp_path / "does_not_exist")


def test_loads_and_recognizes_when_model_files_present(tmp_path):
    model_dir = tmp_path / "korean_handwriting_recognition"
    model_dir.mkdir()
    (model_dir / "inference.pdiparams").touch()

    fake = _FakeTextRecognition([{"rec_text": "홍길동", "rec_score": 0.88}])
    with patch("paddleocr.TextRecognition", return_value=fake) as mock_ctor:
        engine = HandwritingEngine(model_dir)
        items = engine.recognize(np.zeros((10, 20, 3), dtype=np.uint8))

    mock_ctor.assert_called_once_with(
        model_name="korean_PP-OCRv5_mobile_rec", model_dir=str(model_dir)
    )
    assert len(items) == 1
    assert items[0].text == "홍길동"
    assert items[0].confidence == 0.88
    assert items[0].polygon == [[0, 0], [20, 0], [20, 10], [0, 10]]


def test_returns_empty_list_when_recognizer_finds_no_text(tmp_path):
    model_dir = tmp_path / "korean_handwriting_recognition"
    model_dir.mkdir()
    (model_dir / "inference.pdiparams").touch()

    fake = _FakeTextRecognition([{"rec_text": "", "rec_score": 0.0}])
    with patch("paddleocr.TextRecognition", return_value=fake):
        engine = HandwritingEngine(model_dir)
        assert engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8)) == []
