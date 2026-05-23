"""Image preprocessing shared by training and inference."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def preprocess_image(img: np.ndarray, image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize and normalize an OpenCV BGR image for the CNN model."""
    if img is None or img.size == 0:
        raise ValueError("Cannot preprocess an empty image.")

    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    salt_pepper_filtered = cv2.medianBlur(img, 3)
    gaussian_filtered = cv2.GaussianBlur(salt_pepper_filtered, (5, 5), 0)
    uniform_filtered = cv2.medianBlur(gaussian_filtered, 3)
    resized = cv2.resize(uniform_filtered, image_size)
    return resized.astype("float32") / 255.0
