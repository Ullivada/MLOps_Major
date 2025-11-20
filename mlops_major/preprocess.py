from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


IMAGE_SIZE: Tuple[int, int] = (64, 64)


def image_bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert uploaded image bytes to the flattened vector expected by the model."""
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("L")
        img = img.resize(IMAGE_SIZE)
        array = np.asarray(img, dtype=np.float32) / 255.0
        return array.flatten()

