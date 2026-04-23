"""Image loading and preprocessing helpers."""

from __future__ import annotations

from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps


def load_rgb_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def save_rgb_image_temp(image: np.ndarray) -> str:
    array = np.asarray(image)
    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(array.astype(np.uint8)).save(tmp.name)
    return tmp.name


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(float)
    else:
        rgb = image.astype(float)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if gray.max(initial=0) > 1.0:
        gray = gray / 255.0
    return gray
