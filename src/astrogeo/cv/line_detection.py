"""Building line detection with OpenCV LSD/Hough fallback."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from astrogeo.pipeline.result_schema import BuildingLineDetectionResult, LineSegment, bounded

from .preprocess import to_grayscale_float


def _angle_distance_to_vertical(angle_rad: float) -> float:
    return abs(math.atan2(math.sin(angle_rad - math.pi / 2.0), math.cos(angle_rad - math.pi / 2.0)))


def _line_midpoint_inside_mask(line: LineSegment, mask: np.ndarray) -> bool:
    midpoint = line.midpoint
    y = int(round(midpoint.y))
    x = int(round(midpoint.x))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def detect_building_lines(image: np.ndarray, sky_mask: np.ndarray) -> BuildingLineDetectionResult:
    try:
        import cv2
    except Exception:
        return BuildingLineDetectionResult(
            line_segments=[],
            candidate_vertical_lines=[],
            confidence=0.0,
            warnings=["OpenCV unavailable; building line detection skipped."],
            diagnostics={"failure": "opencv_unavailable"},
        )

    gray = to_grayscale_float(image)
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    building_mask = ~np.asarray(sky_mask, dtype=bool)
    if building_mask.shape != gray.shape:
        building_mask = np.ones_like(gray, dtype=bool)
    edges = cv2.Canny(gray_u8, 60, 180)
    edges = (edges * building_mask.astype(np.uint8)).astype(np.uint8)
    height, width = gray.shape
    min_len = max(18.0, 0.035 * min(width, height))

    lines: List[LineSegment] = []
    try:
        detector = cv2.createLineSegmentDetector()
        detected = detector.detect(gray_u8)[0]
        if detected is not None:
            for item in detected.reshape(-1, 4):
                line = LineSegment(float(item[0]), float(item[1]), float(item[2]), float(item[3]))
                if line.length >= min_len and _line_midpoint_inside_mask(line, building_mask):
                    lines.append(line)
    except Exception:
        detected = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=int(min_len),
            maxLineGap=8,
        )
        if detected is not None:
            for item in detected.reshape(-1, 4):
                line = LineSegment(float(item[0]), float(item[1]), float(item[2]), float(item[3]))
                if _line_midpoint_inside_mask(line, building_mask):
                    lines.append(line)

    vertical_candidates = [
        line for line in lines if _angle_distance_to_vertical(line.angle_rad) <= math.radians(40)
    ]
    if len(vertical_candidates) < 3:
        vertical_candidates = sorted(lines, key=lambda line: line.length, reverse=True)[: max(3, len(lines))]

    confidence = bounded(min(len(vertical_candidates) / 12.0, 1.0) * 0.8 + min(len(lines) / 50.0, 1.0) * 0.2)
    warnings = []
    if len(vertical_candidates) < 3:
        warnings.append("Too few candidate vertical building lines detected.")
    return BuildingLineDetectionResult(
        line_segments=lines,
        candidate_vertical_lines=vertical_candidates,
        confidence=float(confidence),
        warnings=warnings,
        diagnostics={
            "line_count": len(lines),
            "candidate_vertical_count": len(vertical_candidates),
            "min_length_px": min_len,
        },
    )

