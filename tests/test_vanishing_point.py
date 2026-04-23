import numpy as np

from astrogeo.cv.vanishing_point import (
    estimate_vertical_vanishing_point,
    line_intersection_homogeneous,
)
from astrogeo.pipeline.result_schema import LineSegment


def test_parallel_vertical_lines_produce_vanishing_point_at_infinity() -> None:
    lines = [
        LineSegment(10, 10, 10, 100),
        LineSegment(30, 20, 30, 120),
        LineSegment(50, 5, 50, 80),
    ]

    result = estimate_vertical_vanishing_point(lines, (150, 100), min_inliers=3)

    assert result.success
    assert result.is_at_infinity
    assert len(result.inlier_lines) == 3


def test_converging_lines_estimate_finite_external_vanishing_point() -> None:
    vp = np.array([50.0, -200.0])
    bases = [(10, 100), (30, 110), (70, 115), (90, 105)]
    lines = []
    for x, y in bases:
        start = np.array([x, y], dtype=float)
        direction = vp - start
        direction = direction / np.linalg.norm(direction)
        end = start + direction * 80.0
        lines.append(LineSegment(start[0], start[1], end[0], end[1]))

    result = estimate_vertical_vanishing_point(lines, (130, 100), min_inliers=4)
    point = result.finite_point()

    assert result.success
    assert point is not None
    assert abs(point.x - vp[0]) < 1e-6
    assert abs(point.y - vp[1]) < 1e-6


def test_line_intersection_uses_homogeneous_coordinates() -> None:
    a = LineSegment(0, 0, 10, 10)
    b = LineSegment(0, 10, 10, 0)

    point = line_intersection_homogeneous(a, b)

    np.testing.assert_allclose(point / point[2], np.array([5.0, 5.0, 1.0]), atol=1e-12)

