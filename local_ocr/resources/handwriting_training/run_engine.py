#!/usr/bin/env python3
"""train_config.yaml을 실행하는 진입점.

설치된 `paddlex` pip 패키지의 CLI(`python -m paddlex ...`)는 학습/평가/
내보내기를 노출하지 않는다 — `python -m paddlex --help`에는 파이프라인
예측·서빙·paddle2onnx 옵션만 있다. 실제 학습/평가/내보내기 경로는
`paddlex.engine.Engine`이 맡는데, 이건 다음 3줄짜리 진입 스크립트가 있어야
커맨드라인에서 쓸 수 있다 (파이프 소스에서 실제로 확인함:
`paddlex/utils/config.py`의 `parse_args()`가 `-c/--config`와 `-o/--override`를
받고, `paddlex/engine.py`의 `Engine`이 `Global.mode`(check_dataset/train/
evaluate/export/predict)에 따라 분기한다).

사용 예:
    python run_engine.py -c train_config.yaml
    python run_engine.py -c train_config.yaml -o Global.mode=train
"""

from paddlex.engine import Engine

if __name__ == "__main__":
    Engine().run()
