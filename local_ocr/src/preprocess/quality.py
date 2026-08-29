"""불확실한 Crop에 어떤 보정을 시도할지 정하기 위한 저비용 판별기.

문서 "이미지 전처리" 표의 "적용 조건" 열(저대비 영역, 역상 텍스트 등)을
간단한 통계로 근사한다. 완벽한 감지가 목적이 아니라, 불필요한 보정본을
줄여 CPU 비용을 아끼는 것이 목적이다 (문서 "CPU 성능 최적화").
"""

from __future__ import annotations

import numpy as np

LOW_CONTRAST_STD_THRESHOLD = 40.0
INVERTED_BORDER_MEAN_THRESHOLD = 128.0


def is_low_contrast(gray: np.ndarray) -> bool:
    return float(gray.std()) < LOW_CONTRAST_STD_THRESHOLD


def is_likely_inverted(gray: np.ndarray) -> bool:
    """가장자리(배경으로 간주)가 어두우면 흰 글자/검은 배경(역상)으로 본다."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float(border.mean()) < INVERTED_BORDER_MEAN_THRESHOLD
