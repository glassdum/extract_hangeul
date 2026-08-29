# 한국어 손글씨 Fine-tuning (Stage 4)

문서 "한국어 손글씨 모델 개발" 항목의 실행 스크립트 모음이다.
`korean_PP-OCRv5_mobile_rec_handwriting_ft`는 공개 모델의 공식 이름이
아니라, **이 스크립트들로 실제로 학습해야 나오는 결과물**이다.

## 이 저장소를 개발한 환경에서 못 하는 것

- AI Hub 데이터 다운로드: 이 환경의 네트워크 정책이 aihub.or.kr을 막고
  있고, 애초에 AI Hub 데이터는 각자 계정으로 이용 신청을 해야 받을 수
  있다. **여기서는 대신 사용자가 직접 신청·다운로드해야 한다:**
  - [AI Hub 대용량 손글씨 OCR 데이터](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=realm&currMenu=100&dataSetSn=605&topMenu=)
  - [AI Hub 다양한 형태의 한글 문자 OCR](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=91)
  - 신청 시 **재배포 조건을 반드시 확인**한다 (문서 "위험 요소와 대응":
    "학습 데이터·가중치 조건 확인 필요 → License 문서 포함·사전 검토").
- 실제 GPU 학습: `train_config.yaml`의 `Global.device: gpu:0`이 그대로
  가리키듯, Fine-tuning 자체는 GPU가 있는 별도 환경에서 사용자가 돌려야
  한다.
- 공식 사전 학습 가중치·문자 사전 다운로드: `pretrain_weight_path`가
  가리키는 bcebos.com도 이 환경에서는 막혀 있다.

## 이 저장소에서 이미 확인해 둔 것

아래 4개 스크립트는 GPU·AI Hub 데이터 없이도 검증 가능한 부분까지 실제로
돌려서 확인했다 (지어낸 명령어가 아니다):

- `prepare_dataset.py`가 만드는 `train.txt`/`val.txt`/`test.txt`/`dict.txt`/
  `images/` 구조가 설치된 **paddlex==3.7.2**의 실제 데이터셋 검사기
  (`check_dataset` 모드)를 통과하는 것까지 확인했다. (마지막에 통계 그래프용
  폰트 파일을 내려받으려다 네트워크 정책에 막혀 멈췄는데, 이는 데이터셋
  형식이 아니라 이 샌드박스만의 인터넷 제한 때문이다 — 실제 학습 PC에서는
  통과한다.)
- `run_engine.py -c train_config.yaml`이 `Global.mode`(check_dataset/
  train/evaluate/export)에 따라 올바르게 분기하는 것도 함께 확인했다.
- `export_to_onnx.py`가 감싸는 `python -m paddlex --paddle2onnx ...` 옵션은
  설치된 paddlex의 `--help` 출력에서 실제로 존재함을 확인했다.
- `evaluate.py`(CER/Exact Match 계산)는 외부 모델이 필요 없어 순수 로직만
  단위 테스트로 검증했다 (`tests/test_evaluate.py`).

## 실행 순서

### 1. AI Hub 원본 → canonical manifest

AI Hub 데이터셋마다 라벨링 JSON 스키마가 다르므로, 이 저장소는 원본을
직접 읽지 않는다. 먼저 아래 형식의 JSONL manifest로 변환하는 작은 스크립트를
데이터셋에 맞게 작성한다 (원본 JSON의 실제 필드명을 봐야 하므로 이건
사용자가 직접 해야 하는 부분이다):

```jsonl
{"image_path": "001.png", "text": "홍길동", "writer_id": "w0001"}
{"image_path": "002.png", "text": "010-1234-5678", "writer_id": "w0002"}
```

- `writer_id`가 없는(인쇄체 등) 데이터는 `""`로 둔다.
- 이 프로젝트 앱을 실제로 쓰면서 사용자가 [판독 불가]를 교정한 Crop도
  같은 manifest에 섞어 넣을 수 있다 (문서 "학습 데이터").

### 2. Train/Val/Test 분할 + PaddleOCR 형식 변환

```bash
python prepare_dataset.py manifest.jsonl \
    --images-root /data/aihub/handwriting/images \
    --output-dir train_data/rec \
    --dict-path /path/to/official_korean_dict.txt \
    --val-ratio 0.1 --test-ratio 0.1
```

작성자(writer_id) 단위로 나눠 같은 사람의 필체가 여러 집합에 섞이지 않게
한다 (문서 "학습 원칙"). `--dict-path`를 꼭 지정한다 — 공식
`korean_PP-OCRv5_mobile_rec` 사전 학습 가중치를 내려받으면 함께 딸려오는
문자 사전 파일이다. 지정하지 않으면 manifest의 글자만으로 새 사전을
만드는데, 그러면 사전 학습 가중치와 호환되지 않아 Fine-tuning이 아니라
처음부터 학습하는 셈이 된다.

### 3. 데이터셋 검사

```bash
python run_engine.py -c train_config.yaml   # Global.mode: check_dataset (기본값)
```

먼저 이걸로 이미지 경로·라벨·사전이 올바른지 확인한다. `train_config.yaml`의
`Global.dataset_dir`을 2단계의 `--output-dir`로 맞춰 둔다.

### 4. 학습

```bash
python run_engine.py -c train_config.yaml -o Global.mode=train
```

GPU가 있는 환경에서 실행한다. `Train.pretrain_weight_path`가 공식
`korean_PP-OCRv5_mobile_rec` 가중치를 가리키므로 그 지점부터 Fine-tuning된다
(문서 "초기 가중치"). 증강(Blur/Noise/Perspective/LowContrast/Compression,
문서 "학습 원칙")은 `train_config.yaml`에 직접 노출돼 있지 않다 — paddlex가
내부적으로 호출하는 PaddleOCR 저장소의 기본 RecAug가 적용되며, 세밀한 조정이
필요하면 PaddleOCR GitHub 저장소를 따로 받아 그 안의 저수준 yaml을 고쳐야
한다.

### 5. 평가

```bash
python run_engine.py -c train_config.yaml -o Global.mode=evaluate
# 학습이 끝나면 test.txt(2단계에서 학습에 전혀 쓰지 않은 홀드아웃)로 직접 추론해
# predictions.txt(이미지경로\t예측)를 만든 다음:
python evaluate.py train_data/rec/test.txt predictions.txt
```

`Evaluate` 모드는 PaddleX가 자동으로 나누는 val 세트 기준이고, `test.txt`는
문서가 요구하는 "개발 중 조정에 사용하지 않은" 완전히 별도의 세트다 —
둘 다 확인하는 것이 좋다.

### 6. ONNX로 내보내기 (CPU 배포용)

```bash
python export_to_onnx.py \
    --paddle-model-dir output/best_accuracy/inference \
    --onnx-model-dir ../models/korean_handwriting_recognition \
    --opset-version 17
```

변환 후에는 같은 이미지를 원본 Paddle 추론과 ONNX Runtime 추론에 각각
넣어 결과가 같은지 검증한다 (문서 "모델별 변환 및 출력 일치 검증").

### 7. 파이프라인에 연결

`../models/korean_handwriting_recognition/`에 결과물을 두면
`src/recognition/handwriting_engine.HandwritingEngine`이 그 경로에서 바로
로드한다. Stage 3(`app.py`)의 앙상블 로직에 이 엔진을 실제로 연결하는
작업은 아직 하지 않았다 — 진짜 가중치가 없는 상태에서 배선을 미리 넣으면
검증할 수 없는 코드가 파이프라인에 섞이기 때문이다. 가중치가 생기면
`recognition.tesseract_engine`을 앙상블에 연결한 패턴(Stage 3 커밋 참고)을
그대로 따라 붙이면 된다.
