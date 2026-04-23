"""Report placeholders for later phases."""

from __future__ import annotations

from iona.pipeline.result_schema import IonaResult


def result_summary_text(result: IonaResult) -> str:
    if not result.success or result.estimated_location is None:
        return "Iona failed: " + ", ".join(result.failure_reasons)
    loc = result.estimated_location
    return f"lat={loc.latitude_deg:.6f}, lon={loc.longitude_deg:.6f}, confidence={result.confidence}"

