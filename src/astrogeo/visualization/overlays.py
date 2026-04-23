"""Debug overlay rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from astrogeo.pipeline.result_schema import (
    BuildingLineDetectionResult,
    SkyMaskResult,
    StarDetectionResult,
    VanishingPointResult,
)


def _draw_line(draw: ImageDraw.ImageDraw, line, fill, width: int = 2) -> None:
    draw.line((line.x1, line.y1, line.x2, line.y2), fill=fill, width=width)


def save_debug_overlay(
    image_path: str,
    output_path: str,
    sky: Optional[SkyMaskResult] = None,
    stars: Optional[StarDetectionResult] = None,
    lines: Optional[BuildingLineDetectionResult] = None,
    vp: Optional[VanishingPointResult] = None,
) -> None:
    base = Image.open(image_path).convert("RGBA")
    width, height = base.size
    if sky and sky.sky_mask is not None:
        mask = np.asarray(sky.sky_mask, dtype=bool)
        if mask.shape == (height, width):
            overlay = np.zeros((height, width, 4), dtype=np.uint8)
            overlay[mask] = np.array([40, 120, 255, 70], dtype=np.uint8)
            base = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    if stars:
        for point in stars.star_candidates[:400]:
            r = 2
            draw.ellipse((point.x - r, point.y - r, point.x + r, point.y + r), outline=(255, 235, 60, 255))

    if lines:
        inlier_ids = set()
        if vp:
            inlier_ids = {id(line) for line in vp.inlier_lines}
        for line in lines.line_segments:
            _draw_line(draw, line, (255, 140, 30, 180), width=1)
        for line in lines.candidate_vertical_lines:
            color = (50, 255, 90, 255) if id(line) in inlier_ids else (255, 80, 60, 220)
            _draw_line(draw, line, color, width=3)

    if vp and vp.vanishing_point_homogeneous is not None:
        point = vp.finite_point()
        if point and 0 <= point.x < width and 0 <= point.y < height:
            r = 8
            draw.ellipse((point.x - r, point.y - r, point.x + r, point.y + r), outline=(255, 0, 0, 255), width=3)
        else:
            vector = vp.vanishing_point_homogeneous
            if abs(float(vector[2])) > 1e-9:
                target = np.array([vector[0] / vector[2], vector[1] / vector[2]], dtype=float)
                origin = np.array([width / 2.0, height / 2.0])
                direction = target - origin
            else:
                origin = np.array([width / 2.0, height / 2.0])
                direction = np.array([vector[0], vector[1]], dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                end = origin + direction * min(width, height) * 0.35
                draw.line((origin[0], origin[1], end[0], end[1]), fill=(255, 0, 0, 255), width=4)
                draw.text((end[0] + 5, end[1] + 5), "zenith/nadir candidate ray", fill=(255, 0, 0, 255))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(output_path)
