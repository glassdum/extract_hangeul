# local_ocr

CPU 기반 완전 로컬 범용 OCR 프로그램. 이미지·PDF에서 한글·영어·숫자(인쇄체와
손글씨)를 인식해 줄바꿈 없는 텍스트(TXT)와 검증용 상세 데이터(JSON)를
만든다. 전체 요구사항·모델 구성·개발 단계는 프로젝트 기술개발계획서를
따른다 (`TECHNICAL DEVELOPMENT PLAN — CPU 기반 완전 로컬 범용 OCR 프로그램
개발 계획서`).

## 현재 구현 범위 (Stage 1~2: 입력·기본 OCR, 전처리 탐색)

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
- confidence 임계값 기반 상태 근사치(`auto_confirmed` /
  `review_required` / `low_confidence`) — 진짜 앙상블 판정 이전의 잠정치
- **Stage 2**: confidence가 낮은 Crop만 골라 Deskew·CLAHE/Contrast
  Stretch·Denoise·Otsu 이진화·2배 확대·반전 등 여러 전처리 보정본을
  만들고, 같은 PaddleOCR 엔진으로 재인식해 confidence가 가장 높은 후보를
  채택한다(`preprocess.variants` + `ensemble.reprocess`). 시도한 모든
  후보는 JSON의 `candidates`에 그대로 남는다.
- TXT(최종 문자열)·JSON(페이지·bbox·후보·상태) 저장

### 아직 없음 (다음 단계)

| 단계 | 내용 |
| --- | --- |
| 3 | 인쇄체/손글씨/Tesseract **엔진 간** 결과 교차 비교, 진짜 "판독 불가" 판정 |
| 4 | 한국어 손글씨 Fine-tuning 모델(`korean_PP-OCRv5_mobile_rec_handwriting_ft`) |
| 5 | Bounding Box 간격 기반 정밀 띄어쓰기 판정 |
| 6 | PySide6 검토 GUI (원본 Crop·후보·수정·판독 불가 처리) |
| 7 | ONNX Runtime 변환, Hash 캐시, Batch/Thread 튜닝 |
| 8 | Windows 오프라인 배포 패키징 |

Perspective Transform(문서 펴기, UVDoc)은 아직 구현하지 않았고, 표 선
제거(`preprocess.lines.remove_table_lines`)는 함수만 있고 기본 보정본
목록에는 자동으로 포함하지 않는다 (`src/preprocess/__init__.py` 참고).

## 요구사항

- Python 3.12.x (Windows 10/11 x64, GPU 불필요)
- 의존성은 `pyproject.toml` 참고. Stage 1 스모크 테스트에 실제로 쓰인
  버전은 `requirements.lock`에 고정돼 있다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.lock

python app.py path/to/input.pdf --output-dir output
python app.py path/to/images_folder --output-dir output --dpi 300 --mode accuracy
```

첫 실행 시 PaddleOCR가 모델 가중치를 huggingface.co / bcebos.com /
modelscope.cn / aistudio.baidu.com 중 하나에서 자동 다운로드해 캐시한다
(완전 오프라인 배포판 패키징은 Stage 8에서 다룬다). **이 저장소를 개발한
샌드박스 환경은 위 호스트로 나가는 연결을 정책상 차단하고 있어 실제 모델
다운로드·추론 자체는 여기서 실행해보지 못했다.** 대신 파이프라인 배선
(입력 판별 → PDF 텍스트 레이어/혼합/스캔 처리 → 읽기 순서 정렬 → TXT/JSON
저장)은 가짜 인식 엔진(`OCREngine` 인터페이스 구현체)으로 `tests/`에서
전부 검증했다. 인터넷이 열려 있는 PC(개발 PC 등)에서 `python app.py`를
실행하면 실제 PaddleOCR 추론까지 포함한 전체 경로가 그대로 동작한다.
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
│  ├─ detection/        # (Stage 3에서 채워짐)
│  ├─ recognition/      # PaddleOCR 엔진 래퍼
│  ├─ ensemble/         # Crop별 전처리 보정본 재판독·최적 후보 선택 (엔진 간 비교는 Stage 3)
│  ├─ spacing/          # 읽기 순서 정렬 + 줄 연결
│  ├─ review/           # (Stage 6에서 채워짐, PySide6)
│  ├─ storage/          # TXT·JSON 저장
│  └─ common/           # 설정, 공용 타입, confidence 근사 로직
├─ resources/           # models/ tessdata/ licenses/ (자산은 Git에 커밋하지 않음)
├─ tests/
└─ build/
```

## 테스트

```bash
pytest
```
