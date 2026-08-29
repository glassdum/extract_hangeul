#!/usr/bin/env python3
"""AI Hub 손글씨/한글 문자 OCR 데이터를 PaddleOCR 인식기 학습 형식으로 변환한다.

이 스크립트는 AI Hub가 배포하는 원본 JSON을 직접 읽지 않는다. AI Hub의
"대용량 손글씨 OCR 데이터"(dataSetSn=605)와 "다양한 형태의 한글 문자
OCR"(dataSetSn=91)은 라벨링 JSON 스키마가 서로 다르고, 이 프로젝트를 만든
환경은 AI Hub 다운로드가 네트워크 정책상 막혀 있어 실제 파일로 스키마를
확인할 수 없었다 — 확인 못 한 스키마를 코드로 잘못 단정하느니, 검증 가능한
경계에서 인터페이스를 끊는 편이 안전하다고 판단했다.

대신 아래 canonical manifest(JSONL, 한 줄에 레코드 하나)를 입력으로 받는다.
AI Hub 원본 → 이 manifest 변환은 데이터셋별로 별도 스크립트를 작성해야
한다 (원본 필드명이 데이터셋마다 다르므로).

    {"image_path": "raw/001.png", "text": "홍길동", "writer_id": "w0001"}
    {"image_path": "raw/002.png", "text": "010-1234-5678", "writer_id": "w0002"}

- image_path: --images-root 기준 상대 경로. PaddleX는 학습 시 이미지 경로를
  dataset_dir(= --output-dir) 기준으로 찾으므로(실제로 확인함 — 다르면
  DatasetFileNotFoundError), 이 스크립트가 이미지를 --images-root에서
  --output-dir 아래 같은 상대 경로로 복사해 둔다.
- text: 정답 문자열 (이 프로젝트에서 사용자가 교정한 Crop 라벨도 같은
  manifest에 섞어 넣을 수 있다 — 문서 "학습 데이터": "실제 프로그램에서
  사용자가 수정한 Crop 이미지와 정답 문자열")
- writer_id: 작성자 식별자. 인쇄체처럼 "작성자"가 의미 없는 데이터는 빈
  문자열("")로 두면 된다 — 그런 레코드는 사람 단위 분리 대상에서 빠지고
  항상 train에만 들어간다 (문서 "인쇄체 성능이 급격히 떨어지지 않도록
  인쇄체와 손글씨 데이터를 함께 Fine-tuning한다").

핵심 로직은 문서 "학습 원칙"의 첫 항목이다: "Train·Validation·Test
작성자를 분리해 같은 사람의 필체가 여러 집합에 섞이지 않도록 한다." —
샘플 단위가 아니라 writer_id 단위로 무작위 분할한다.

출력 파일명(train.txt/val.txt/dict.txt)은 설치된 paddlex==3.7.2의
`TextRecConfig.update_dataset()`이 기본으로 찾는 이름 그대로다 (소스 직접
확인: `paddlex/repo_apis/PaddleOCR_api/text_rec/config.py`) — 이 형식이
`train_config.yaml`과 그대로 맞물리도록 맞췄다. test.txt는 PaddleX가
자동으로 쓰지 않는 이 프로젝트만의 홀드아웃 세트로, evaluate.py가 학습이
끝난 뒤 별도로 쓴다 (문서 "개발 중 조정에 사용하지 않은 Test Set을 별도로
유지한다").

dict.txt(문자 사전)는 이 스크립트가 함부로 새로 만들지 않는다. 손글씨
Fine-tuning은 공식 korean_PP-OCRv5_mobile_rec 체크포인트에서 시작하는데
(문서 "초기 가중치"), 그 체크포인트는 이미 고정된 사전으로 학습돼 있어
사전을 바꾸면 기존 가중치와 맞지 않게 된다. `--dict-path`로 그 공식
사전 파일(인터넷이 열린 PC에서 사전 학습 모델을 내려받으면 함께 딸려
온다)을 지정하는 것이 정석이다. 지정하지 않으면 이 manifest에 나온
글자만으로 사전을 새로 만드는데, 이는 사전 학습 가중치와 호환되지
않으므로 처음부터 새로 학습할 때만 쓰라는 경고를 함께 출력한다.

사용 예:
    python prepare_dataset.py manifest.jsonl \
        --output-dir train_data/rec \
        --dict-path /path/to/official_korean_dict.txt \
        --val-ratio 0.1 --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image_path: str
    text: str
    writer_id: str


def load_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                samples.append(
                    Sample(
                        image_path=record["image_path"],
                        text=record["text"],
                        writer_id=record.get("writer_id", ""),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no}: 필수 필드 누락 ({exc})") from None
    return samples


def split_by_writer(
    samples: list[Sample], val_ratio: float, test_ratio: float, seed: int = 42
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """작성자(writer_id) 단위로 train/val/test를 나눠 필체 유출을 막는다.

    writer_id가 없는 샘플(인쇄체 등)은 항상 train에 넣는다 — "작성자"라는
    개념이 없는 데이터를 억지로 나누는 것은 의미가 없고, 인쇄체 성능
    유지에는 오히려 train에 많이 들어가는 편이 낫다.
    """
    with_writer: dict[str, list[Sample]] = {}
    without_writer: list[Sample] = []
    for sample in samples:
        if sample.writer_id:
            with_writer.setdefault(sample.writer_id, []).append(sample)
        else:
            without_writer.append(sample)

    writer_ids = sorted(with_writer.keys())
    random.Random(seed).shuffle(writer_ids)

    total = len(writer_ids)
    n_test = round(total * test_ratio)
    n_val = round(total * val_ratio)
    test_writers = set(writer_ids[:n_test])
    val_writers = set(writer_ids[n_test : n_test + n_val])
    train_writers = set(writer_ids[n_test + n_val :])

    train = list(without_writer) + [s for wid in train_writers for s in with_writer[wid]]
    val = [s for wid in val_writers for s in with_writer[wid]]
    test = [s for wid in test_writers for s in with_writer[wid]]
    return train, val, test


def copy_images(samples: list[Sample], images_root: Path, output_dir: Path) -> None:
    """PaddleX가 dataset_dir(=output_dir) 기준 상대 경로로 이미지를 찾으므로,
    원본 위치가 어디든 output_dir 아래 같은 상대 경로에 실제로 복사해 둔다."""
    for sample in samples:
        src = images_root / sample.image_path
        dest = output_dir / sample.image_path
        if dest.resolve() == src.resolve():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)


def write_label_file(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            # PaddleOCR 인식기 학습 라벨 형식: "이미지경로\t정답문자열"
            f.write(f"{sample.image_path}\t{sample.text}\n")


def write_dict_file(path: Path, samples: list[Sample], dict_path: Path | None) -> None:
    if dict_path is not None:
        shutil.copyfile(dict_path, path)
        return

    print(
        "경고: --dict-path 없이 manifest의 글자만으로 dict.txt를 새로 만듭니다. "
        "공식 사전 학습 가중치로 Fine-tuning할 계획이라면 그 가중치가 쓴 사전을 "
        "--dict-path로 지정하세요 (사전이 다르면 Fine-tuning이 되지 않습니다)."
    )
    chars: set[str] = set()
    for sample in samples:
        chars.update(sample.text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ch in sorted(chars):
            f.write(f"{ch}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="canonical manifest (JSONL) 경로")
    parser.add_argument("--output-dir", type=Path, default=Path("train_data/rec"))
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="manifest의 image_path가 상대 경로로 가리키는 기준 디렉터리 (기본: manifest가 있는 폴더)",
    )
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=None,
        help="공식 korean_PP-OCRv5_mobile_rec가 쓰는 문자 사전 파일 (권장)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    samples = load_manifest(args.manifest)
    if not samples:
        print("manifest에 유효한 레코드가 없습니다.")
        return 1

    train, val, test = split_by_writer(samples, args.val_ratio, args.test_ratio, args.seed)

    images_root = args.images_root or args.manifest.resolve().parent
    copy_images(samples, images_root, args.output_dir)

    write_label_file(args.output_dir / "train.txt", train)
    write_label_file(args.output_dir / "val.txt", val)
    write_label_file(args.output_dir / "test.txt", test)
    write_dict_file(args.output_dir / "dict.txt", samples, args.dict_path)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
