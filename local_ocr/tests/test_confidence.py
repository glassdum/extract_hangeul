from common.config import PipelineConfig
from common.confidence import classify_status


def test_classify_status_thresholds():
    config = PipelineConfig(auto_confirm_threshold=0.8, review_threshold=0.5)
    assert classify_status(0.95, config) == "auto_confirmed"
    assert classify_status(0.8, config) == "auto_confirmed"
    assert classify_status(0.79, config) == "review_required"
    assert classify_status(0.5, config) == "review_required"
    assert classify_status(0.49, config) == "low_confidence"
