"""검토 세션: PySide6 없이 테스트 가능한 상태·로직 계층.

`review.main_window`의 Qt 화면은 이 클래스를 감싸는 얇은 껍데기일 뿐이고,
JSON 로딩·원본 Crop 재생성·수정 반영·저장·이력 기록은 전부 여기 있다.

`storage.writer.save_json`이 만드는 JSON은 문서의 JSON 예시와 같은 모양
(text/status/page/bbox/candidates)이라 confidence·source를 항목마다 따로
저장하지 않는다 — 그래서 이 클래스는 TextLine으로 역직렬화하지 않고 원본
dict를 그대로 들고 있다가 필요한 부분만 고쳐서 되쓴다(값을 지어내지
않기 위해).
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from common.types import BBox, TextLine, UNREADABLE_MARKER
from input.loader import detect_input_type
from preprocess.crop import crop_with_padding
from spacing.reading_order import join_text
from storage.history import CorrectionRecord, HistoryStore
from storage.writer import save_txt

NEEDS_REVIEW_STATUSES = {"review_required", "low_confidence", "unreadable"}


@dataclass
class ReviewItem:
    page: int
    bbox: BBox
    text: str
    status: str
    candidates: dict[str, tuple[str, float]]
    payload: dict  # 원본 JSON에서 이 줄에 해당하는 dict (in-place로 갱신됨)


class ReviewSession:
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        self._payload: dict = json.loads(self.json_path.read_text(encoding="utf-8"))
        self._pending_history: list[CorrectionRecord] = []

    @property
    def source_path(self) -> str:
        return self._payload["source"]

    @property
    def final_text(self) -> str:
        return self._payload.get("final_text", "")

    def all_items(self) -> list[ReviewItem]:
        return [
            self._to_item(line_data)
            for page_data in self._payload["pages"]
            for line_data in page_data["lines"]
        ]

    def items_needing_review(self) -> list[ReviewItem]:
        return [item for item in self.all_items() if item.status in NEEDS_REVIEW_STATUSES]

    def apply_correction(self, item: ReviewItem, text: str, status: str = "auto_confirmed") -> None:
        """사용자가 후보를 고르거나 직접 고친 텍스트를 반영한다."""
        text = unicodedata.normalize("NFC", text.strip())
        self._pending_history.append(
            CorrectionRecord(
                source_path=self.source_path,
                page=item.page,
                bbox=tuple(item.bbox.as_list()),
                before_text=item.payload["text"],
                before_status=item.payload["status"],
                after_text=text,
                after_status=status,
            )
        )
        item.payload["text"] = text
        item.payload["status"] = status
        item.text, item.status = text, status

    def mark_unreadable(self, item: ReviewItem) -> None:
        self.apply_correction(item, UNREADABLE_MARKER, status="unreadable")

    def get_crop_image(self, item: ReviewItem, pad_ratio: float = 0.15) -> Image.Image:
        """원본 소스 파일에서 이 줄의 bbox에 해당하는 이미지를 다시 잘라낸다."""
        page_image = self._load_page_image(item.page)
        return crop_with_padding(page_image, item.bbox, pad_ratio=pad_ratio)

    def save(self, history_db_path: str | Path | None = None) -> None:
        """JSON·TXT를 갱신하고, 이력 DB 경로가 주어지면 이번 세션의 수정 내역도 남긴다."""
        lines = [
            TextLine(
                page=line_data["page"],
                bbox=BBox(*line_data["bbox"]),
                text=line_data["text"],
                confidence=1.0,
                source="reviewed",
                status=line_data["status"],
            )
            for page_data in self._payload["pages"]
            for line_data in page_data["lines"]
        ]
        self._payload["final_text"] = join_text(lines)

        self.json_path.write_text(
            json.dumps(self._payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_txt(self.json_path.with_suffix(".txt"), self._payload["final_text"])

        if history_db_path is not None and self._pending_history:
            store = HistoryStore(history_db_path)
            for record in self._pending_history:
                store.add(record)
        self._pending_history.clear()

    def export_correction_to_manifest(
        self, item: ReviewItem, images_dir: str | Path, manifest_path: str | Path
    ) -> None:
        """문서 "학습 데이터": "실제 프로그램에서 사용자가 수정한 Crop 이미지와
        정답 문자열"을 Stage 4 `prepare_dataset.py`가 바로 읽을 수 있는 manifest
        레코드로 남긴다. writer_id는 알 수 없으므로 비워 둔다(그러면
        prepare_dataset.py가 항상 train에만 넣는다)."""
        images_dir = Path(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        crop = self.get_crop_image(item, pad_ratio=0.0)
        stem = Path(self.source_path).stem
        image_name = f"{stem}_p{item.page}_{int(item.bbox.x0)}_{int(item.bbox.y0)}.png"
        crop.save(images_dir / image_name)

        record = {"image_path": image_name, "text": item.text, "writer_id": ""}
        manifest_path = Path(manifest_path)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _to_item(self, line_data: dict) -> ReviewItem:
        candidates = {
            key: (value[0], value[1]) for key, value in line_data.get("candidates", {}).items()
        }
        return ReviewItem(
            page=line_data["page"],
            bbox=BBox(*line_data["bbox"]),
            text=line_data["text"],
            status=line_data["status"],
            candidates=candidates,
            payload=line_data,
        )

    def _load_page_image(self, page_no: int) -> Image.Image:
        source = Path(self.source_path)
        kind = detect_input_type(source)

        if kind == "image":
            from input.image_loader import load_image_frames

            frames = load_image_frames(source)
            frame = next((f for f in frames if f.index == page_no), frames[0])
            return frame.image

        return _render_pdf_page(source, page_no, target_width=self._page_width(page_no))

    def _page_width(self, page_no: int) -> int:
        for page_data in self._payload["pages"]:
            if page_data["page"] == page_no:
                return page_data["width"]
        raise ValueError(f"JSON에 페이지 {page_no} 정보가 없습니다.")


def _render_pdf_page(source: Path, page_no: int, target_width: int) -> Image.Image:
    import pymupdf

    with pymupdf.open(str(source)) as doc:
        page = doc[page_no - 1]
        zoom = target_width / page.rect.width
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False
        )
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
