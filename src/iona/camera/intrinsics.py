"""Pinhole camera intrinsics estimation."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from iona.pipeline.result_schema import CameraIntrinsics, ExifInfo, PlateSolveResult


def _focal_pixels_from_35mm(width: int, height: int, focal_35mm: float) -> float:
    sensor_diag_mm = 43.266
    image_diag_px = math.hypot(width, height)
    fov_diag = 2.0 * math.atan(sensor_diag_mm / (2.0 * focal_35mm))
    return (image_diag_px / 2.0) / math.tan(fov_diag / 2.0)


def estimate_camera_intrinsics(
    image_shape: Tuple[int, int],
    exif_info: Optional[ExifInfo] = None,
    plate_result: Optional[PlateSolveResult] = None,
) -> CameraIntrinsics:
    height, width = image_shape[:2]
    warnings = []
    source = "default"
    confidence = 0.35

    focal_px = None
    if plate_result and plate_result.pixel_scale_arcsec:
        radians_per_pixel = plate_result.pixel_scale_arcsec / 206265.0
        if radians_per_pixel > 0:
            focal_px = 1.0 / radians_per_pixel
            source = "plate_scale"
            confidence = 0.75

    if focal_px is None and exif_info and exif_info.focal_length_35mm:
        focal_px = _focal_pixels_from_35mm(width, height, exif_info.focal_length_35mm)
        source = "exif_35mm"
        confidence = 0.65

    if focal_px is None:
        focal_px = max(width, height) * 1.2
        warnings.append("Camera focal length unavailable; using a conservative pinhole estimate.")

    if exif_info and exif_info.focal_length_35mm and exif_info.focal_length_35mm < 24:
        confidence *= 0.75
        warnings.append("Wide-angle lens detected; distortion is not corrected in this MVP.")

    return CameraIntrinsics(
        fx=float(focal_px),
        fy=float(focal_px),
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
        confidence=float(confidence),
        warnings=warnings,
        source=source,
    )

