"""Automatic sky mask estimation using lightweight image statistics."""

from __future__ import annotations

from typing import List

import numpy as np

from iona.pipeline.result_schema import SkyMaskResult, bounded

from .preprocess import to_grayscale_float


def _fallback_sky_mask(image: np.ndarray) -> SkyMaskResult:
    gray = to_grayscale_float(image)
    height, width = gray.shape
    mask = np.zeros_like(gray, dtype=bool)
    mask[: max(1, height // 2), :] = True
    return SkyMaskResult(
        sky_mask=mask,
        confidence=0.20,
        sky_fraction=float(np.mean(mask)),
        warnings=["OpenCV unavailable; using top-half fallback sky mask."],
        diagnostics={"method": "top_half_fallback", "image_shape": [height, width]},
    )


def estimate_sky_mask(image: np.ndarray) -> SkyMaskResult:
    try:
        import cv2
    except Exception:
        return _fallback_sky_mask(image)

    gray = to_grayscale_float(image)
    height, width = gray.shape
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = cv2.blur((edges > 0).astype(np.float32), (31, 31))

    dark_threshold = float(np.quantile(gray, 0.72))
    low_edge_threshold = float(np.quantile(edge_density, 0.55))
    dark = gray <= dark_threshold
    low_edge = edge_density <= low_edge_threshold
    top_bias = np.linspace(1.0, 0.35, height)[:, None]
    mask = dark & low_edge & (top_bias > 0.45)

    kernel = np.ones((7, 7), np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    kept = np.zeros_like(mask_u8, dtype=bool)
    image_area = height * width
    largest_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        top = int(stats[label, cv2.CC_STAT_TOP])
        if area < image_area * 0.03:
            continue
        if top > height * 0.65:
            continue
        kept |= labels == label
        largest_area = max(largest_area, area)

    warnings: List[str] = []
    if not np.any(kept):
        warnings.append("No stable sky component found; falling back to low-edge dark regions.")
        kept = mask_u8.astype(bool)

    sky_fraction = float(np.mean(kept))
    area_score = 1.0 - min(abs(sky_fraction - 0.35) / 0.35, 1.0)
    connectedness = float(largest_area / max(1, np.count_nonzero(kept)))
    confidence = bounded(0.15 + 0.55 * area_score + 0.30 * connectedness)
    if sky_fraction < 0.08 or sky_fraction > 0.85:
        confidence *= 0.5
        warnings.append("Sky mask area is outside the expected range.")

    return SkyMaskResult(
        sky_mask=kept,
        confidence=float(confidence),
        sky_fraction=sky_fraction,
        warnings=warnings,
        diagnostics={
            "method": "dark_low_edge_components",
            "dark_threshold": dark_threshold,
            "low_edge_threshold": low_edge_threshold,
            "component_count": int(num_labels - 1),
            "largest_component_area": int(largest_area),
        },
    )

