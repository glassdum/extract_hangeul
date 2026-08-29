#!/usr/bin/env python3
"""PaddlePaddle 추론 모델을 ONNX로 변환한다.

문서: "학습은 GPU 환경에서 수행할 수 있지만 배포 모델은 ONNX로 변환해
CPU에서 실행한다."

설치된 paddlex(paddleocr의 의존성)가 제공하는 `--paddle2onnx` CLI를 그대로
감싼 것이다 — `python -m paddlex --help`의 "Paddle2ONNX Options"에서 실제
옵션(`--paddle_model_dir` / `--onnx_model_dir` / `--opset_version`)을
확인하고 만들었다. paddle2onnx 플러그인이 설치돼 있지 않다면 먼저 설치한다:

    python -m paddlex --install paddle2onnx

사용 예:
    python export_to_onnx.py \
        --paddle-model-dir ./output/handwriting_ft/inference \
        --onnx-model-dir ../models/korean_handwriting_recognition \
        --opset-version 17

변환 후 출력 검증(문서 "모델별 변환 및 출력 일치 검증")은 별도로 해야 한다:
같은 이미지를 원본 Paddle 추론과 ONNX Runtime 추론에 각각 넣어 결과가
같은지 비교한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--paddle-model-dir", required=True, help="학습 결과 추론 모델 디렉터리")
    parser.add_argument("--onnx-model-dir", required=True, help="ONNX 출력 디렉터리")
    parser.add_argument("--opset-version", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cmd = [
        sys.executable,
        "-m",
        "paddlex",
        "--paddle2onnx",
        "--paddle_model_dir",
        args.paddle_model_dir,
        "--onnx_model_dir",
        args.onnx_model_dir,
        "--opset_version",
        str(args.opset_version),
    ]
    print("실행:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
