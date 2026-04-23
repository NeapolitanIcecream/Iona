"""Coordinate conversion helpers."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def normalize_angle_360(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def normalize_angle_180(angle_deg: float) -> float:
    value = normalize_angle_360(angle_deg + 180.0) - 180.0
    if value == -180.0:
        return 180.0
    return float(value)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return arr / norm


def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)],
        dtype=float,
    )


def unit_vector_to_radec(vector: np.ndarray) -> Tuple[float, float]:
    x, y, z = normalize_vector(vector)
    ra = math.degrees(math.atan2(y, x))
    dec = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    return normalize_angle_360(ra), float(dec)

