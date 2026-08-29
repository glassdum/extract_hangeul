"""PySide6 검토 화면 (문서 "검토 GUI": 원본 Crop, 후보, 수정, 판독 불가 처리).

로직은 전부 `review.session.ReviewSession`에 있다 — 이 파일은 그 위에 얹은
얇은 화면일 뿐이라, 여기서 하는 일은 "위젯을 그리고 세션 메서드를 부르는"
정도로 최소화했다.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .session import ReviewItem, ReviewSession


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())  # data 버퍼 수명과 분리하기 위해 copy()


class ReviewWindow(QMainWindow):
    def __init__(self, json_path: str | Path, history_db_path: str | Path | None = None):
        super().__init__()
        self.session = ReviewSession(json_path)
        self.history_db_path = history_db_path
        self._items: list[ReviewItem] = []

        self.setWindowTitle(f"OCR 검토 - {Path(json_path).name}")
        self._build_ui()
        self._reload_items()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        root.addWidget(self.list_widget, 1)

        right = QVBoxLayout()

        self.image_label = QLabel("Crop 없음")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(150)
        right.addWidget(self.image_label)

        right.addWidget(QLabel("후보 (더블클릭하면 아래 텍스트에 채워짐)"))
        self.candidates_list = QListWidget()
        self.candidates_list.itemDoubleClicked.connect(self._on_candidate_double_clicked)
        right.addWidget(self.candidates_list)

        right.addWidget(QLabel("최종 텍스트"))
        self.text_edit = QLineEdit()
        right.addWidget(self.text_edit)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("적용")
        self.apply_button.clicked.connect(self._on_apply)
        self.unreadable_button = QPushButton("판독 불가로 표시")
        self.unreadable_button.clicked.connect(self._on_mark_unreadable)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.unreadable_button)
        right.addLayout(button_row)

        self.save_button = QPushButton("저장")
        self.save_button.clicked.connect(self._on_save)
        right.addWidget(self.save_button)

        self.status_label = QLabel()
        right.addWidget(self.status_label)

        root.addLayout(right, 1)
        self.setCentralWidget(central)

    def _reload_items(self) -> None:
        self._items = self.session.items_needing_review()
        self.list_widget.clear()
        for item in self._items:
            self.list_widget.addItem(f"[{item.status}] p{item.page}: {item.text}")
        self.status_label.setText(f"검토 필요: {len(self._items)}건")
        if self._items:
            self.list_widget.setCurrentRow(0)
        else:
            self._on_selection_changed(-1)

    def _current_item(self) -> ReviewItem | None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def _on_selection_changed(self, _row: int) -> None:
        item = self._current_item()
        if item is None:
            self.image_label.setText("Crop 없음")
            self.candidates_list.clear()
            self.text_edit.clear()
            return

        crop = self.session.get_crop_image(item)
        self.image_label.setPixmap(pil_to_pixmap(crop))

        self.candidates_list.clear()
        for key, (text, confidence) in item.candidates.items():
            list_item = QListWidgetItem(f"{key}: {text} ({confidence:.2f})")
            list_item.setData(Qt.ItemDataRole.UserRole, text)
            self.candidates_list.addItem(list_item)

        self.text_edit.setText(item.text)

    def _on_candidate_double_clicked(self, list_item: QListWidgetItem) -> None:
        self.text_edit.setText(list_item.data(Qt.ItemDataRole.UserRole))

    def _on_apply(self) -> None:
        item = self._current_item()
        if item is None:
            return
        self.session.apply_correction(item, self.text_edit.text())
        self._reload_items()

    def _on_mark_unreadable(self) -> None:
        item = self._current_item()
        if item is None:
            return
        self.session.mark_unreadable(item)
        self._reload_items()

    def _on_save(self) -> None:
        self.session.save(history_db_path=self.history_db_path)
        QMessageBox.information(self, "저장 완료", f"{self.session.json_path} 저장했습니다.")
