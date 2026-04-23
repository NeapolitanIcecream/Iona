"""Vertical vanishing point estimation with homogeneous coordinates."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from astrogeo.pipeline.result_schema import LineSegment, VanishingPointResult, bounded


def line_intersection_homogeneous(line_a: LineSegment, line_b: LineSegment) -> np.ndarray:
    point = np.cross(line_a.homogeneous_line(), line_b.homogeneous_line())
    norm = np.linalg.norm(point)
    if norm == 0:
        return point
    point = point / norm
    if abs(point[2]) > 1e-12 and point[2] < 0:
        point = -point
    return point


def _line_direction(line: LineSegment) -> np.ndarray:
    direction = np.array([line.x2 - line.x1, line.y2 - line.y1], dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def angular_residual_rad(line: LineSegment, vp_h: np.ndarray) -> float:
    direction = _line_direction(line)
    if np.linalg.norm(direction) == 0:
        return math.pi / 2.0
    if abs(float(vp_h[2])) > 1e-9:
        vp_xy = np.array([vp_h[0] / vp_h[2], vp_h[1] / vp_h[2]], dtype=float)
        midpoint = line.midpoint
        target = vp_xy - np.array([midpoint.x, midpoint.y], dtype=float)
    else:
        target = np.array([vp_h[0], vp_h[1]], dtype=float)
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        return math.pi / 2.0
    target = target / target_norm
    dot = abs(float(np.clip(np.dot(direction, target), -1.0, 1.0)))
    return float(math.acos(dot))


def residual_px(line: LineSegment, vp_h: np.ndarray) -> float:
    return float(math.sin(angular_residual_rad(line, vp_h)) * max(line.length, 1.0))


def refine_vanishing_point(lines: Sequence[LineSegment]) -> Optional[np.ndarray]:
    if len(lines) < 2:
        return None
    rows = []
    for line in lines:
        homogeneous = line.homogeneous_line()
        if np.linalg.norm(homogeneous[:2]) == 0:
            continue
        rows.append(homogeneous * max(line.length, 1.0))
    if len(rows) < 2:
        return None
    matrix = np.vstack(rows)
    _, _, vt = np.linalg.svd(matrix)
    point = vt[-1, :]
    norm = np.linalg.norm(point)
    if norm == 0:
        return None
    point = point / norm
    if abs(point[2]) > 1e-12 and point[2] < 0:
        point = -point
    return point


def estimate_vertical_vanishing_point(
    line_segments: Sequence[LineSegment],
    image_shape: Tuple[int, int],
    max_trials: int = 250,
    inlier_angle_deg: float = 5.0,
    min_inliers: int = 3,
    random_seed: int = 17,
) -> VanishingPointResult:
    usable = [line for line in line_segments if line.length >= 8]
    if len(usable) < 2:
        return VanishingPointResult(
            success=False,
            vanishing_point_homogeneous=None,
            inlier_lines=[],
            residual_px=None,
            confidence=0.0,
            failure_reason="not_enough_lines",
            diagnostics={"line_count": len(usable)},
        )

    rng = np.random.default_rng(random_seed)
    threshold = math.radians(inlier_angle_deg)
    best_vp = None
    best_inliers: List[LineSegment] = []
    best_score = float("inf")
    trials = min(max_trials, max(1, len(usable) * (len(usable) - 1) // 2))

    for _ in range(trials):
        idx = rng.choice(len(usable), size=2, replace=False)
        candidate = line_intersection_homogeneous(usable[int(idx[0])], usable[int(idx[1])])
        if np.linalg.norm(candidate) == 0:
            continue
        inliers = [line for line in usable if angular_residual_rad(line, candidate) <= threshold]
        if len(inliers) < 2:
            continue
        score = float(np.median([residual_px(line, candidate) for line in inliers]))
        if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and score < best_score):
            best_vp = candidate
            best_inliers = inliers
            best_score = score

    if best_vp is None:
        return VanishingPointResult(
            success=False,
            vanishing_point_homogeneous=None,
            inlier_lines=[],
            residual_px=None,
            confidence=0.0,
            failure_reason="ransac_no_consensus",
            diagnostics={"line_count": len(usable), "trials": trials},
        )

    refined_candidate = refine_vanishing_point(best_inliers)
    refined = refined_candidate if refined_candidate is not None else best_vp
    residuals = [residual_px(line, refined) for line in best_inliers]
    median_residual = float(np.median(residuals)) if residuals else None
    inlier_ratio = len(best_inliers) / max(1, len(usable))
    residual_score = 1.0 if median_residual is None else 1.0 - min(median_residual / 12.0, 1.0)
    confidence = bounded(0.55 * inlier_ratio + 0.45 * residual_score)
    warnings: List[str] = []
    if len(best_inliers) < min_inliers:
        return VanishingPointResult(
            success=False,
            vanishing_point_homogeneous=refined,
            inlier_lines=best_inliers,
            residual_px=median_residual,
            confidence=confidence * 0.5,
            warnings=["Vanishing point consensus has too few inlier lines."],
            failure_reason="not_enough_vanishing_point_inliers",
            diagnostics={"line_count": len(usable), "inlier_count": len(best_inliers)},
        )

    if abs(float(refined[2])) < 1e-9:
        warnings.append("Vertical vanishing point is at infinity or extremely far outside the image.")
    else:
        x, y = refined[0] / refined[2], refined[1] / refined[2]
        height, width = image_shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            warnings.append("Vertical vanishing point is outside the image.")

    return VanishingPointResult(
        success=True,
        vanishing_point_homogeneous=refined,
        inlier_lines=best_inliers,
        residual_px=median_residual,
        confidence=confidence,
        warnings=warnings,
        diagnostics={
            "line_count": len(usable),
            "inlier_count": len(best_inliers),
            "inlier_ratio": inlier_ratio,
            "trials": trials,
        },
    )
