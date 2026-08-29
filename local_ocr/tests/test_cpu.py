from unittest.mock import patch

from common.cpu import recommended_thread_count


def test_recommended_thread_count_reserves_one_core_by_default():
    with patch("common.cpu.os.cpu_count", return_value=8):
        assert recommended_thread_count() == 7


def test_recommended_thread_count_never_goes_below_one():
    with patch("common.cpu.os.cpu_count", return_value=1):
        assert recommended_thread_count() == 1
    with patch("common.cpu.os.cpu_count", return_value=2):
        assert recommended_thread_count(reserve=4) == 1


def test_recommended_thread_count_falls_back_to_one_when_cpu_count_unknown():
    with patch("common.cpu.os.cpu_count", return_value=None):
        assert recommended_thread_count() == 1
