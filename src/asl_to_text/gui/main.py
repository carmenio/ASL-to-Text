"""PySide6 desktop application entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from asl_to_text.hands import HandExtractor
from asl_to_text.predictor import SignPredictor

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    QApplication = None
    QFrame = None
    QHBoxLayout = None
    QImage = None
    QLabel = None
    QMainWindow = object
    QPixmap = None
    QPushButton = None
    QTimer = None
    QVBoxLayout = None
    QWidget = None
    Qt = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ASL-to-Text desktop app.")
    parser.add_argument("--model", default="models/best_model_CNN.keras", help="Path to the Keras model.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    return parser


class MainWindow(QMainWindow):
    def __init__(self, model_path: str, camera_index: int) -> None:
        if QApplication is None:
            raise RuntimeError("PySide6 is required for the desktop app. Run `pip install -e .`.")

        super().__init__()
        self.setWindowTitle("ASL-to-Text")
        self.resize(1120, 720)
        self.camera_index = camera_index
        self.cap = None
        self.extractor = None
        self.predictor = SignPredictor(model_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.video_label = QLabel("Camera is stopped")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 480)
        self.video_label.setObjectName("video")

        self.prediction_label = QLabel("Setup mode")
        self.prediction_label.setObjectName("prediction")
        self.confidence_label = QLabel("Confidence: --")
        self.model_label = QLabel(self.predictor.status_message)
        self.camera_label = QLabel(f"Camera: {camera_index}")
        self.status_label = QLabel("Start the camera to begin.")
        self.status_label.setWordWrap(True)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(QLabel("Prediction"))
        panel_layout.addWidget(self.prediction_label)
        panel_layout.addWidget(self.confidence_label)
        panel_layout.addSpacing(20)
        panel_layout.addWidget(QLabel("Model"))
        panel_layout.addWidget(self.model_label)
        panel_layout.addSpacing(20)
        panel_layout.addWidget(QLabel("Input"))
        panel_layout.addWidget(self.camera_label)
        panel_layout.addWidget(self.status_label)
        panel_layout.addStretch()
        panel_layout.addLayout(controls)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.video_label, 3)
        layout.addWidget(panel, 1)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow { background: #f6f8fb; }
            QLabel { color: #17202a; font-size: 15px; }
            QLabel#video {
                background: #111827;
                color: #d1d5db;
                border-radius: 6px;
                font-size: 22px;
            }
            QFrame#panel {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 16px;
            }
            QLabel#prediction {
                color: #0f766e;
                font-size: 48px;
                font-weight: 700;
            }
            QPushButton {
                background: #1d4ed8;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:disabled { background: #94a3b8; }
            """
        )

    def start_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_label.setText(f"Could not open webcam index {self.camera_index}.")
            self.video_label.setText("Camera unavailable")
            self.cap.release()
            self.cap = None
            return

        try:
            self.extractor = HandExtractor(static_image_mode=False)
        except RuntimeError as exc:
            self.status_label.setText(str(exc))
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Camera running. Hold one hand in frame.")
        self.timer.start(30)

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.extractor is not None:
            self.extractor.close()
            self.extractor = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.setText("Camera is stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self) -> None:
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_label.setText("Failed to read from webcam.")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        crop = self.extractor.extract(frame, draw=True) if self.extractor else None
        prediction = self.predictor.predict(crop.image if crop else None)
        self.prediction_label.setText(prediction.label)
        self.confidence_label.setText(
            f"Confidence: {prediction.confidence:.0%}" if prediction.model_ready else "Confidence: --"
        )
        self.model_label.setText(self.predictor.status_message)
        if prediction.message:
            self.status_label.setText(prediction.message)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(rgb_frame.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self.stop_camera()
        event.accept()


def run_app(model_path: str, camera_index: int) -> int:
    if QApplication is None:
        raise RuntimeError("PySide6 is required for the desktop app. Run `pip install -e .`.")

    app = QApplication(sys.argv)
    window = MainWindow(str(Path(model_path)), camera_index)
    window.show()
    return app.exec()


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(run_app(args.model, args.camera))


if __name__ == "__main__":
    main()
