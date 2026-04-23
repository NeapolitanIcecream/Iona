from datetime import datetime, timezone

from astrogeo.astronomy.coordinates import normalize_angle_180
from astrogeo.astronomy.geolocation import (
    estimate_location_from_zenith,
    longitude_error_from_time_seconds,
    zenith_radec_from_location,
)


def test_zenith_radec_round_trips_to_known_location() -> None:
    utc = datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
    ra, dec = zenith_radec_from_location(35.72, 139.81, utc)

    location = estimate_location_from_zenith(ra, dec, utc, estimated_time_error_seconds=2)

    assert abs(location.latitude_deg - 35.72) < 1e-9
    assert abs(normalize_angle_180(location.longitude_deg - 139.81)) < 1e-9
    assert location.estimated_lon_error_deg == longitude_error_from_time_seconds(2)


def test_longitude_wraps_to_signed_degrees() -> None:
    utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
    location = estimate_location_from_zenith(1.0, -12.0, utc)

    assert -180.0 <= location.longitude_deg <= 180.0


def test_one_minute_time_error_maps_to_quarter_degree_longitude_error() -> None:
    assert longitude_error_from_time_seconds(60) == 0.25

