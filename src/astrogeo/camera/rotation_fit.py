"""Fit camera-to-celestial rotation from matched rays."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from astrogeo.astronomy.coordinates import normalize_vector, radec_to_unit_vector
from astrogeo.camera.rays import image_point_to_camera_ray
from astrogeo.pipeline.result_schema import CameraIntrinsics, RotationFitResult


def fit_rotation_kabsch(camera_vectors: np.ndarray, celestial_vectors: np.ndarray) -> RotationFitResult:
    camera = np.asarray(camera_vectors, dtype=float)
    celestial = np.asarray(celestial_vectors, dtype=float)
    if camera.shape != celestial.shape or camera.ndim != 2 or camera.shape[1] != 3:
        return RotationFitResult(
            success=False,
            rotation_matrix=None,
            residual_deg=None,
            sample_count=0,
            confidence=0.0,
            failure_reason="invalid_vector_shapes",
        )
    if camera.shape[0] < 2:
        return RotationFitResult(
            success=False,
            rotation_matrix=None,
            residual_deg=None,
            sample_count=int(camera.shape[0]),
            confidence=0.0,
            failure_reason="not_enough_rotation_samples",
        )

    camera = np.array([normalize_vector(vec) for vec in camera])
    celestial = np.array([normalize_vector(vec) for vec in celestial])
    covariance = camera.T @ celestial
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    predicted = (rotation @ camera.T).T
    dots = np.clip(np.sum(predicted * celestial, axis=1), -1.0, 1.0)
    residual_deg = float(np.degrees(np.mean(np.arccos(dots))))
    confidence = max(0.0, min(1.0, 1.0 - residual_deg / 5.0))
    return RotationFitResult(
        success=True,
        rotation_matrix=rotation,
        residual_deg=residual_deg,
        sample_count=int(camera.shape[0]),
        confidence=confidence,
        diagnostics={"determinant": float(np.linalg.det(rotation))},
    )


def fit_camera_to_celestial_rotation(
    wcs: object,
    intrinsics: CameraIntrinsics,
    sample_pixels: Sequence[Tuple[float, float]],
) -> RotationFitResult:
    if wcs is None:
        return RotationFitResult(
            success=False,
            rotation_matrix=None,
            residual_deg=None,
            sample_count=0,
            confidence=0.0,
            failure_reason="missing_wcs",
        )
    if len(sample_pixels) < 2:
        return RotationFitResult(
            success=False,
            rotation_matrix=None,
            residual_deg=None,
            sample_count=len(sample_pixels),
            confidence=0.0,
            failure_reason="not_enough_wcs_samples",
        )
    try:
        xs = np.array([pixel[0] for pixel in sample_pixels], dtype=float)
        ys = np.array([pixel[1] for pixel in sample_pixels], dtype=float)
        coords = wcs.pixel_to_world(xs, ys)
        ras = np.asarray(coords.ra.degree, dtype=float)
        decs = np.asarray(coords.dec.degree, dtype=float)
    except Exception as exc:
        return RotationFitResult(
            success=False,
            rotation_matrix=None,
            residual_deg=None,
            sample_count=len(sample_pixels),
            confidence=0.0,
            failure_reason="wcs_pixel_to_world_failed",
            diagnostics={"error": str(exc)},
        )

    camera_vectors: List[np.ndarray] = []
    celestial_vectors: List[np.ndarray] = []
    for x, y, ra, dec in zip(xs, ys, ras, decs):
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        camera_vectors.append(image_point_to_camera_ray(np.array([x, y, 1.0]), intrinsics))
        celestial_vectors.append(radec_to_unit_vector(float(ra), float(dec)))
    return fit_rotation_kabsch(np.array(camera_vectors), np.array(celestial_vectors))

