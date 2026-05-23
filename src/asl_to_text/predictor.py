"""Model loading and prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .labels import CLASS_LABELS
from .preprocessing import preprocess_image


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    model_ready: bool
    message: str = ""


class SignPredictor:
    """Load a Keras model and predict ASL classes from hand crops."""

    def __init__(
        self,
        model_path: str | Path = "models/best_model_CNN.keras",
        labels: Sequence[str] = CLASS_LABELS,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.image_size = image_size
        self.model = None
        self.status_message = "Model not loaded."
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            self.model = None
            self.status_message = f"Model missing: {self.model_path}"
            return

        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            self.model = None
            self.status_message = "TensorFlow is not installed."
            raise RuntimeError("TensorFlow is required to load a Keras model.") from exc

        self.model = load_model(self.model_path)
        self.status_message = f"Model loaded: {self.model_path}"

    def predict(self, hand_image: np.ndarray | None) -> Prediction:
        if hand_image is None or hand_image.size == 0:
            return Prediction("No hand", 0.0, self.is_ready, "No hand detected.")

        if self.model is None:
            return Prediction("Setup mode", 0.0, False, self.status_message)

        processed = preprocess_image(hand_image, self.image_size)
        batch = np.expand_dims(processed, axis=0)
        scores = self.model.predict(batch, verbose=0)[0]
        class_index = int(np.argmax(scores))
        label = self.labels[class_index] if class_index < len(self.labels) else "Unknown"
        confidence = float(scores[class_index])
        return Prediction(label, confidence, True)
