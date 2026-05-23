"""MediaPipe hand detection and crop helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandCrop:
    image: np.ndarray
    box: Tuple[int, int, int, int]


class HandExtractor:
    """Extract a square hand crop from a webcam frame."""

    def __init__(
        self,
        margin: int = 100,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        hand_landmarker_model: str | None = None,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for hand detection. Install the project with `pip install -e .`."
            ) from exc

        self.margin = margin
        self._backend = "legacy"
        self._static_image_mode = static_image_mode
        self._timestamp_ms = int(time.monotonic() * 1000)
        self._mp = mp

        if hasattr(mp, "solutions"):
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        except ImportError as exc:
            raise RuntimeError(
                "This MediaPipe version does not provide the legacy Solutions API or the Tasks hand landmarker."
            ) from exc

        self._backend = "tasks"
        model_path = self._resolve_hand_landmarker_model(hand_landmarker_model)
        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = HandLandmarker.create_from_options(options)

    @staticmethod
    def _resolve_hand_landmarker_model(model_path: str | None) -> str:
        if model_path:
            return model_path

        path = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
        if path.exists():
            return str(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".task.download")
        try:
            with urlopen(HAND_LANDMARKER_MODEL_URL, timeout=60) as response:
                temporary_path.write_bytes(response.read())
            temporary_path.replace(path)
        except (OSError, URLError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            raise RuntimeError(
                "MediaPipe Tasks requires a hand landmarker model. "
                f"Download {HAND_LANDMARKER_MODEL_URL} and save it to {path}."
            ) from exc

        return str(path)

    def extract(self, frame: np.ndarray, draw: bool = True) -> Optional[HandCrop]:
        if frame is None or frame.size == 0:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_landmarks = self._detect_landmarks(rgb_frame)
        if not hand_landmarks:
            return None

        hand_landmarks = hand_landmarks[0]
        height, width = frame.shape[:2]
        x_values = [int(landmark.x * width) for landmark in hand_landmarks.landmark]
        y_values = [int(landmark.y * height) for landmark in hand_landmarks.landmark]

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        square_size = max(crop_width, crop_height) + (2 * self.margin)
        center_x = x_min + crop_width // 2
        center_y = y_min + crop_height // 2

        left = max(0, center_x - square_size // 2)
        top = max(0, center_y - square_size // 2)
        right = min(width, center_x + square_size // 2)
        bottom = min(height, center_y + square_size // 2)

        if left >= right or top >= bottom:
            return None

        if draw:
            cv2.rectangle(frame, (left, top), (right, bottom), (55, 220, 120), 2)

        return HandCrop(image=frame[top:bottom, left:right].copy(), box=(left, top, right, bottom))

    def _detect_landmarks(self, rgb_frame: np.ndarray):
        if self._backend == "legacy":
            results = self._hands.process(rgb_frame)
            return results.multi_hand_landmarks

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        if self._static_image_mode:
            result = self._hands.detect(image)
        else:
            self._timestamp_ms += 1
            result = self._hands.detect_for_video(image, self._timestamp_ms)
        return [_TaskLandmarkList(landmarks) for landmarks in result.hand_landmarks]

    def close(self) -> None:
        self._hands.close()


@dataclass(frozen=True)
class _TaskLandmarkList:
    landmark: list


def extract_hand(frame: np.ndarray, margin: int = 100, draw: bool = True) -> Optional[HandCrop]:
    """Convenience wrapper for one-off hand extraction."""
    extractor = HandExtractor(margin=margin)
    try:
        return extractor.extract(frame, draw=draw)
    finally:
        extractor.close()
