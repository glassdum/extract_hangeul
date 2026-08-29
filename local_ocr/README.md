# local_ocr

CPU 기반 완전 로컬 범용 OCR 프로그램. 이미지·PDF에서 한글·영어·숫자(인쇄체와
손글씨)를 인식해 줄바꿈 없는 텍스트(TXT)와 검증용 상세 데이터(JSON)를
만든다. 전체 요구사항·모델 구성·개발 단계는 프로젝트 기술개발계획서를
따른다 (`TECHNICAL DEVELOPMENT PLAN — CPU 기반 완전 로컬 범용 OCR 프로그램
개발 계획서`).

## 현재 구현 범위 (Stage 1~3 코드 + Stage 4 학습 스캐폴딩)

- 입력 판별: PDF vs 이미지, PDF의 텍스트 레이어 유무 판별
- PDF: 텍스트 레이어 직접 추출(bbox 포함) + 혼합 PDF의 임베디드 이미지
  영역만 OCR + 텍스트 레이어가 없는 스캔 페이지는 전체 렌더링 후 OCR
- 이미지: PNG/JPG/BMP/TIFF/WebP/HEIC/HEIF/GIF/PSD 로딩, EXIF 방향 보정,
  Alpha 합성, 다중 프레임(TIFF/GIF) 분리
- 문서/줄 방향 보정 및 기본 인식: PaddleOCR 통합 파이프라인
  (`use_doc_orientation_classify` + `use_textline_orientation` +
  PP-OCRv5_mobile_det + `korean_PP-OCRv5_mobile_rec`)
- 읽기 순서 정렬(위→아래, 왼쪽→오른쪽) 후 줄바꿈을 공백 하나로 연결,
  연속 공백 정규화, Unicode NFC 정규화
- **Stage 2**: confidence가 낮은 Crop만 골라 Deskew·CLAHE/Contrast
  Stretch·Denoise·Otsu 이진화·2배 확대·반전 등 여러 전처리 보정본을
  만들고, 같은 PaddleOCR 엔진으로 재인식해 confidence가 가장 높은 후보를
  채택한다(`preprocess.variants` + `ensemble.reprocess`).
- **Stage 3**: Stage 2가 고른 Paddle 최적 후보를 Tesseract 5(kor+eng)로
  독립 재판독해 비교한다(`recognition.tesseract_engine` +
  `ensemble.cross_check`). 완전히 일치하면 자동 확정, 일부 글자만 다르면
  그 구간만 `[판독 불가]`로 접어 넣고(문서 예시 "홍[판독 불가]동" 그대로
  재현됨 — 아래 참고), 결과가 근본적으로 다르면 사용자 확인 대상으로
  남긴다. 손글씨 Fine-tuning 모델(Stage 4)이 아직 없어 "복수 모델 일치"는
  Paddle-Tesseract 일치로 근사한다. 시도한 모든 후보(Paddle 변형들 +
  Tesseract)는 JSON의 `candidates`에 그대로 남는다.
- TXT(최종 문자열)·JSON(페이지·bbox·후보·상태) 저장
- **Stage 4 (스캐폴딩만)**: 한국어 손글씨 Fine-tuning은 AI Hub 데이터
  신청·다운로드와 GPU 학습이 필요해 이 저장소를 만든 환경에서는 실제 학습을
  실행할 수 없다. 대신 `resources/handwriting_training/`에 데이터 준비
  (`prepare_dataset.py`, 작성자 단위 train/val/test 분리) → 학습
  (`run_engine.py` + `train_config.yaml`, 공식 사전 학습 가중치에서
  시작) → 평가(`evaluate.py`, CER/Exact Match) → ONNX 변환
  (`export_to_onnx.py`)까지 이어지는 실행 가능한 스크립트를 준비해 뒀고,
  GPU·AI Hub 데이터 없이 검증 가능한 부분(데이터셋 형식, CLI 배선)은 실제로
  돌려서 확인했다 — 자세한 내용은 그 폴더의 README 참고. 학습된 가중치를
  `recognition.handwriting_engine.HandwritingEngine`이 바로 쓸 수 있게
  인터페이스도 준비해 뒀지만, 실제 가중치가 없어 `app.py` 파이프라인에는
  아직 연결하지 않았다.

### 아직 없음 (다음 단계)

| 단계 | 내용 |
| --- | --- |
| 4 (계속) | 실제 AI Hub 데이터로 학습 실행 + `app.py` 앙상블에 연결 (사용자가 GPU 환경에서 직접) |
| 5 | Bounding Box 간격 기반 정밀 띄어쓰기 판정 |
| 6 | PySide6 검토 GUI (원본 Crop·후보·수정·판독 불가 처리) |
| 7 | ONNX Runtime 변환(엔진 자체), Hash 캐시, Batch/Thread 튜닝 |
| 8 | Windows 오프라인 배포 패키징 |

Perspective Transform(문서 펴기, UVDoc)은 아직 구현하지 않았고, 표 선
제거(`preprocess.lines.remove_table_lines`)는 함수만 있고 기본 보정본
목록에는 자동으로 포함하지 않는다 (`src/preprocess/__init__.py` 참고).
Tesseract osd를 이용한 "방향 충돌 시 페이지 방향 교차 확인"도 아직 없다.

## 요구사항

- Python 3.12.x (Windows 10/11 x64, GPU 불필요)
- 시스템에 Tesseract 5 바이너리 + kor/eng 언어 데이터가 설치돼 있어야
  Stage 3 교차 판독이 동작한다 (Ubuntu/Debian:
  `apt install tesseract-ocr tesseract-ocr-kor`; Windows는 공식
  설치본에 언어 데이터를 포함해 함께 배포한다 — 문서 "초기 Python
  패키지 계획"). 설치돼 있지 않으면 `--no-cross-check`로 Stage 2까지만
  실행할 수 있다.
- 나머지 의존성은 `pyproject.toml` 참고. 실제 통합 테스트에 쓰인 버전은
  `requirements.lock`에 고정돼 있다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.lock

python app.py path/to/input.pdf --output-dir output
python app.py path/to/images_folder --output-dir output --dpi 300 --mode accuracy
python app.py path/to/input.pdf --output-dir output --no-cross-check  # Tesseract 미설치 시
```

첫 실행 시 PaddleOCR가 모델 가중치를 huggingface.co / bcebos.com /
modelscope.cn / aistudio.baidu.com 중 하나에서 자동 다운로드해 캐시한다
(완전 오프라인 배포판 패키징은 Stage 8에서 다룬다). **이 저장소를 개발한
샌드박스 환경은 위 호스트로 나가는 연결을 정책상 차단하고 있어 실제
PaddleOCR 모델 다운로드·추론은 여기서 실행해보지 못했다.** 그 부분은
가짜 Paddle 엔진(`OCREngine` 인터페이스 구현체)으로 배선만 `tests/`에서
검증했다. 인터넷이 열려 있는 PC(개발 PC 등)에서 `python app.py`를
실행하면 실제 PaddleOCR 추론까지 포함한 전체 경로가 그대로 동작한다.

**Tesseract(Stage 3)는 시스템 바이너리라 이 샌드박스에도 설치할 수 있었고,
가짜 Paddle 엔진 + 실제 Tesseract 엔진 조합으로 `python app.py`를 직접
실행해 세 갈래(자동 확정/판독 불가/사용자 확인) 모두 확인했다.** 예를 들어
가짜 Paddle 엔진이 "홍깈동"(오탈자)을 낮은 confidence로 반환하도록 하면,
실제 Tesseract가 같은 Crop에서 정확히 "홍길동"을 읽어내고, 두 결과를 비교한
`ensemble.cross_check`가 서로 다른 한 글자만 표시를 접어 넣어 최종 결과가
문서 예시와 똑같은 `홍[판독 불가]동`이 됐다.

결과는 `output/<파일명>.txt`, `output/<파일명>.json`으로 저장된다.

## 프로젝트 구조

```
local_ocr/
├─ app.py                # CLI 진입점 (Stage 1 파이프라인 조립)
├─ pyproject.toml
├─ requirements.lock
├─ src/
│  ├─ input/            # PDF·이미지 로딩
│  ├─ preprocess/       # 정규화 + 전처리 보정본 생성 (Deskew/CLAHE/이진화/확대/반전 등)
│  ├─ detection/        # (아직 비어 있음 — 정밀 검출기 단독 제어용, 필요해지면 채움)
│  ├─ recognition/      # PaddleOCR 엔진 + Tesseract 교차 판독 엔진 + 손글씨 엔진(가중치 없이는 비활성)
│  ├─ ensemble/         # Crop별 전처리 보정본 재판독(Stage 2) + Paddle-Tesseract 교차 판정(Stage 3)
│  ├─ spacing/          # 읽기 순서 정렬 + 줄 연결
│  ├─ review/           # (Stage 6에서 채워짐, PySide6)
│  ├─ storage/          # TXT·JSON 저장
│  └─ common/           # 설정, 공용 타입, confidence 근사 로직
├─ resources/
│  ├─ models/ tessdata/ licenses/  # 자산은 Git에 커밋하지 않음
│  └─ handwriting_training/        # Stage 4 데이터 준비·학습·평가·ONNX 변환 스크립트
├─ tests/
└─ build/
```

## 테스트

```bash
pytest
```
