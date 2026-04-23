"""Camera ray utilities."""

from __future__ import annotations

import numpy as np

from iona.astronomy.coordinates import normalize_vector
from iona.pipeline.result_schema import CameraIntrinsics


def image_point_to_camera_ray(point_h: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    point = np.asarray(point_h, dtype=float)
    if point.shape != (3,):
        raise ValueError("point_h must be a length-3 homogeneous image point.")
    ray = intrinsics.inverse_matrix @ point
    return normalize_vector(ray)

