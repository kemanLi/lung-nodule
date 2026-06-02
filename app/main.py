from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import HistoryDatabase
from app.services.inference_service import InferenceService


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("肺结节检测分割 MVP")
        self.resize(1180, 760)
        self.image_path: str | None = None
        self.service = InferenceService()
        self.db = HistoryDatabase()

        self.image_label = QLabel("选择一张 CT 切片、DICOM 或 mhd 文件")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 640)
        self.image_label.setStyleSheet("QLabel { background: #15181d; color: #c9d1d9; border: 1px solid #30363d; }")

        self.file_label = QLabel("未选择文件")
        self.det_weights = QLineEdit()
        self.det_weights.setPlaceholderText("可选：YOLO 权重路径，例如 runs/detection/.../best.pt")
        self.seg_weights = QLineEdit()
        self.seg_weights.setPlaceholderText("可选：U-Net 权重路径，例如 runs/segmentation/unet_mvp/best.pt")

        pick_btn = QPushButton("选择图像")
        pick_btn.clicked.connect(self.pick_image)
        run_btn = QPushButton("运行检测 + 分割")
        run_btn.clicked.connect(self.run_analysis)
        det_btn = QPushButton("选择检测权重")
        det_btn.clicked.connect(lambda: self.pick_weights(self.det_weights))
        seg_btn = QPushButton("选择分割权重")
        seg_btn.clicked.connect(lambda: self.pick_weights(self.seg_weights))

        controls = QFormLayout()
        controls.addRow("当前图像", self.file_label)
        controls.addRow("检测权重", self._row(self.det_weights, det_btn))
        controls.addRow("分割权重", self._row(self.seg_weights, seg_btn))

        self.detection_list = QListWidget()
        self.detection_list.setMinimumHeight(160)

        self.history = QTableWidget(0, 5)
        self.history.setHorizontalHeaderLabels(["时间", "结节数", "模式", "原图", "结果"])
        self.history.horizontalHeader().setStretchLastSection(True)
        self.refresh_history()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addLayout(controls)
        right_layout.addWidget(pick_btn)
        right_layout.addWidget(run_btn)
        right_layout.addWidget(QLabel("检测结果"))
        right_layout.addWidget(self.detection_list)
        right_layout.addWidget(QLabel("历史记录"))
        right_layout.addWidget(self.history)

        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.image_label)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

    def _row(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        layout.addWidget(button)
        return widget

    def pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CT 图像",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;DICOM (*.dcm);;MetaImage (*.mhd);;All Files (*)",
        )
        if not path:
            return
        self.image_path = path
        self.file_label.setText(path)
        self.show_image(path)

    def pick_weights(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择权重", "", "PyTorch Weights (*.pt *.pth);;All Files (*)")
        if path:
            target.setText(path)

    def show_image(self, path: str) -> None:
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText("非普通图片格式已选择，运行分析后显示叠加结果")

    def run_analysis(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "缺少图像", "请先选择一张 CT 图像。")
            return
        try:
            result = self.service.analyze(self.image_path, self.det_weights.text().strip(), self.seg_weights.text().strip())
        except Exception as exc:
            QMessageBox.critical(self, "运行失败", str(exc))
            return

        overlay = result["overlay_path"]
        self.show_image(overlay)
        self.detection_list.clear()
        for idx, det in enumerate(result["detections"], start=1):
            self.detection_list.addItem(
                f"#{idx} conf={det['confidence']:.2f} box=({det['x1']},{det['y1']},{det['x2']},{det['y2']})"
            )
        self.db.add_study(self.image_path, overlay, len(result["detections"]), result["mode"])
        self.refresh_history()

    def refresh_history(self) -> None:
        rows = self.db.list_studies()
        self.history.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.history.setItem(r, c, QTableWidgetItem(str(value)))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
