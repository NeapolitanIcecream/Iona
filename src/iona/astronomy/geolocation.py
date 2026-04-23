"""Estimate geographic coordinates from a zenith RA/Dec."""

from __future__ import annotations

from datetime import datetime

from iona.pipeline.result_schema import LocationEstimate

from .coordinates import normalize_angle_180, normalize_angle_360
from .sidereal import greenwich_mean_sidereal_time_deg


def estimate_location_from_zenith(
    ra_zenith_deg: float,
    dec_zenith_deg: float,
    utc_time: datetime,
    estimated_time_error_seconds: float = 0.0,
) -> LocationEstimate:
    gmst_deg = greenwich_mean_sidereal_time_deg(utc_time)
    longitude_deg = normalize_angle_180(ra_zenith_deg - gmst_deg)
    lon_error = estimated_time_error_seconds * 15.0 / 3600.0
    return LocationEstimate(
        latitude_deg=float(dec_zenith_deg),
        longitude_deg=float(longitude_deg),
        gmst_deg=float(gmst_deg),
        estimated_lon_error_deg=float(lon_error),
    )


def zenith_radec_from_location(
    latitude_deg: float, longitude_deg: float, utc_time: datetime
) -> tuple:
    gmst_deg = greenwich_mean_sidereal_time_deg(utc_time)
    return normalize_angle_360(gmst_deg + longitude_deg), float(latitude_deg)


def longitude_error_from_time_seconds(seconds: float) -> float:
    return float(seconds) * 15.0 / 3600.0

