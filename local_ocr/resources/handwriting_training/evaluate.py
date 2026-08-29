#!/usr/bin/env python3
"""학습된 손글씨 인식기를 test_list.txt(prepare_dataset.py 산출물)로 평가한다.

CER(Character Error Rate)과 Exact Match만 계산한다. Space Accuracy, False
Accept Rate 등 문서 "시험 및 품질 지표"의 나머지 지표는 검출·앙상블 결과까지
포함한 전체 파이프라인 단위 지표라 이 스크립트(인식기 단독 평가) 범위 밖이다.

라벨 파일과 예측 파일 모두 prepare_dataset.py와 같은 형식이다:
"이미지경로\t문자열". 예측 파일은 학습이 끝나 가중치가 생긴 뒤
`recognition.handwriting_engine.HandwritingEngine`(또는 PaddleOCR 자체
추론 스크립트)로 test_list.txt의 각 이미지를 돌려 미리 만들어 둔다 — 이
스크립트는 두 파일을 비교하는 평가 로직만 맡는다.

사용 예:
    python evaluate.py train_data/rec/test_list.txt predictions.txt
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    """표준 Levenshtein 편집 거리(삽입·삭제·치환 각 비용 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,  # 삭제
                current[j - 1] + 1,  # 삽입
                previous[j - 1] + cost,  # 치환/일치
            )
        previous = current
    return previous[-1]


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def load_label_file(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            image_path, _, text = line.partition("\t")
            labels[image_path] = _normalize(text)
    return labels


def evaluate(references: dict[str, str], hypotheses: dict[str, str]) -> dict[str, float]:
    common_keys = sorted(set(references) & set(hypotheses))
    missing = sorted(set(references) - set(hypotheses))
    if not common_keys:
        raise ValueError("정답과 예측 파일에 공통 이미지 경로가 없습니다.")

    total_edits = 0
    total_ref_chars = 0
    exact_matches = 0

    for key in common_keys:
        ref = references[key]
        hyp = hypotheses[key]
        total_edits += edit_distance(ref, hyp)
        total_ref_chars += max(len(ref), 1)
        if ref == hyp:
            exact_matches += 1

    return {
        "num_samples": len(common_keys),
        "num_missing_predictions": len(missing),
        "cer": total_edits / total_ref_chars,
        "exact_match": exact_matches / len(common_keys),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("labels", type=Path, help="정답 라벨 파일 (이미지경로\\t정답)")
    parser.add_argument("predictions", type=Path, help="예측 결과 파일 (이미지경로\\t예측)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    references = load_label_file(args.labels)
    hypotheses = load_label_file(args.predictions)

    result = evaluate(references, hypotheses)
    print(f"샘플 수:      {result['num_samples']}")
    print(f"예측 누락:    {result['num_missing_predictions']}")
    print(f"CER:          {result['cer']:.4f}")
    print(f"Exact Match:  {result['exact_match']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
