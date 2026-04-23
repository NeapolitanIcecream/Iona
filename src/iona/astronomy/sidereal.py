"""Sidereal-time helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from .coordinates import normalize_angle_360


def _julian_date(utc_time: datetime) -> float:
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    dt = utc_time.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day = dt.day
    hour = (
        dt.hour
        + dt.minute / 60.0
        + (dt.second + dt.microsecond / 1_000_000.0) / 3600.0
    )
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    jd_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd_day + hour / 24.0


def greenwich_mean_sidereal_time_deg(utc_time: datetime) -> float:
    """Return GMST in degrees for a UTC datetime.

    Astropy is used when installed. The fallback is the common Meeus expression,
    accurate enough for this MVP's rule-based error budget.
    """

    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    utc_time = utc_time.astimezone(timezone.utc)
    try:
        from astropy.utils import iers
        from astropy.time import Time

        iers.conf.auto_download = False
        return float(Time(utc_time).sidereal_time("mean", "greenwich").degree)
    except Exception:
        jd = _julian_date(utc_time)
        t = (jd - 2451545.0) / 36525.0
        gmst = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * t * t
            - (t**3) / 38710000.0
        )
        return normalize_angle_360(gmst)
