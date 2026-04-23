"""Lightweight star candidate detection inside a sky mask."""

from __future__ import annotations

from typing import List

import numpy as np

from astrogeo.pipeline.result_schema import Point, StarDetectionResult, bounded

from .preprocess import to_grayscale_float


def detect_star_candidates(image: np.ndarray, sky_mask: np.ndarray) -> StarDetectionResult:
    gray = to_grayscale_float(image)
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape != gray.shape or not np.any(mask):
        return StarDetectionResult(
            star_candidates=[],
            star_count=0,
            star_density=0.0,
            confidence=0.0,
            warnings=["Sky mask is empty or incompatible with image shape."],
            diagnostics={"failure": "invalid_sky_mask"},
        )

    sky_values = gray[mask]
    threshold = max(
        float(np.mean(sky_values) + 2.5 * np.std(sky_values)),
        float(np.quantile(sky_values, 0.992)),
    )
    bright = (gray >= threshold) & mask

    try:
        import cv2

        bright_u8 = bright.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_u8, 8)
        points: List[Point] = []
        rejected_large = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 1 or area > 80:
                rejected_large += int(area > 80)
                continue
            x, y = centroids[label]
            points.append(Point(float(x), float(y)))
    except Exception:
        ys, xs = np.nonzero(bright)
        points = [Point(float(x), float(y)) for x, y in zip(xs[:500], ys[:500])]
        rejected_large = 0

    sky_area = max(1, int(np.count_nonzero(mask)))
    density = len(points) / (sky_area / 1_000_000.0)
    confidence = bounded(len(points) / 40.0)
    warnings: List[str] = []
    if len(points) < 12:
        warnings.append("Few star candidates detected; plate solving may fail.")
    return StarDetectionResult(
        star_candidates=points,
        star_count=len(points),
        star_density=float(density),
        confidence=float(confidence),
        warnings=warnings,
        diagnostics={
            "threshold": threshold,
            "sky_area_px": sky_area,
            "rejected_large_components": int(rejected_large),
        },
    )

