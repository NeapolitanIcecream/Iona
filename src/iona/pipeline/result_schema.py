"""Dataclasses shared by the Iona pipeline.

The schema intentionally keeps diagnostics machine-readable so the CLI JSON can
explain failure paths without asking the user to annotate the image.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if getattr(value, field.name) is not None
        }
    if isinstance(value, np.ndarray):
        if value.ndim > 1 and value.size > 200:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nonzero": int(np.count_nonzero(value)),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_homogeneous(self) -> np.ndarray:
        return np.array([self.x, self.y, 1.0], dtype=float)


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    strength: float = 1.0

    @property
    def length(self) -> float:
        return float(math.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def midpoint(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def angle_rad(self) -> float:
        return float(math.atan2(self.y2 - self.y1, self.x2 - self.x1))

    def homogeneous_line(self) -> np.ndarray:
        p1 = np.array([self.x1, self.y1, 1.0], dtype=float)
        p2 = np.array([self.x2, self.y2, 1.0], dtype=float)
        line = np.cross(p1, p2)
        norm = np.linalg.norm(line[:2])
        if norm == 0:
            return line
        return line / norm


@dataclass
class ExifInfo:
    capture_time_raw: Optional[str] = None
    offset_time_raw: Optional[str] = None
    focal_length_mm: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    lens_model: Optional[str] = None
    camera_model: Optional[str] = None
    orientation: Optional[int] = None
    gps_present_ignored: bool = False
    raw_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkyMaskResult:
    sky_mask: Optional[np.ndarray]
    confidence: float
    sky_fraction: float
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StarDetectionResult:
    star_candidates: List[Point]
    star_count: int
    star_density: float
    confidence: float
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneMaskResult:
    sky: SkyMaskResult
    building_mask: Optional[np.ndarray]
    backend: str
    model_id: Optional[str]
    confidence: float
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def sky_mask(self) -> Optional[np.ndarray]:
        return self.sky.sky_mask


@dataclass
class PlateSolveResult:
    success: bool
    wcs_header: Optional[Dict[str, Any]] = None
    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None
    pixel_scale_arcsec: Optional[float] = None
    orientation_deg: Optional[float] = None
    matched_stars: Optional[int] = None
    residual_arcsec: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_wcs(self) -> Any:
        if not self.success or not self.wcs_header:
            return None
        try:
            from astropy.io.fits import Header
            from astropy.wcs import WCS
        except Exception:
            return None
        header = Header()
        for key, value in self.wcs_header.items():
            keyword = str(key).strip()
            if keyword.upper() in {"", "COMMENT", "HISTORY", "END"}:
                continue
            try:
                header[keyword] = value
            except (TypeError, ValueError):
                continue
        return WCS(header)


@dataclass
class BuildingLineDetectionResult:
    line_segments: List[LineSegment]
    candidate_vertical_lines: List[LineSegment]
    confidence: float
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VanishingPointResult:
    success: bool
    vanishing_point_homogeneous: Optional[np.ndarray]
    inlier_lines: List[LineSegment]
    residual_px: Optional[float]
    confidence: float
    warnings: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_at_infinity(self) -> bool:
        if self.vanishing_point_homogeneous is None:
            return False
        return abs(float(self.vanishing_point_homogeneous[2])) < 1e-9

    def finite_point(self) -> Optional[Point]:
        if self.vanishing_point_homogeneous is None or self.is_at_infinity:
            return None
        x, y, w = self.vanishing_point_homogeneous
        return Point(float(x / w), float(y / w))


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    confidence: float
    warnings: List[str] = field(default_factory=list)
    source: str = "unknown"

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @property
    def inverse_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)


@dataclass
class RotationFitResult:
    success: bool
    rotation_matrix: Optional[np.ndarray]
    residual_deg: Optional[float]
    sample_count: int
    confidence: float
    warnings: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZenithEstimate:
    success: bool
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    zenith_vector: Optional[np.ndarray]
    selected_sign: Optional[int]
    positive_altitude_fraction: Optional[float]
    confidence: float
    warnings: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocationEstimate:
    latitude_deg: float
    longitude_deg: float
    gmst_deg: float
    estimated_lat_error_deg: Optional[float] = None
    estimated_lon_error_deg: Optional[float] = None


@dataclass
class PipelineEvent:
    stage: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IonaResult:
    success: bool
    estimated_location: Optional[LocationEstimate]
    confidence: str
    quality: Dict[str, Any]
    warnings: List[str]
    failure_reasons: List[str]
    diagnostics: List[PipelineEvent]
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_json(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")


def confidence_label(score: float, hard_failed: bool = False) -> str:
    if hard_failed:
        return "failed"
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    if score >= 0.20:
        return "low"
    return "failed"


def bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
