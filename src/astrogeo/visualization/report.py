"""Report placeholders for later phases."""

from __future__ import annotations

from astrogeo.pipeline.result_schema import AstroGeoResult


def result_summary_text(result: AstroGeoResult) -> str:
    if not result.success or result.estimated_location is None:
        return "AstroGeo failed: " + ", ".join(result.failure_reasons)
    loc = result.estimated_location
    return f"lat={loc.latitude_deg:.6f}, lon={loc.longitude_deg:.6f}, confidence={result.confidence}"

