"""Rule-based quality aggregation."""

from __future__ import annotations

from typing import Iterable

from astrogeo.pipeline.result_schema import bounded, confidence_label


def aggregate_confidence(scores: Iterable[float], hard_failed: bool = False) -> str:
    values = [bounded(score) for score in scores]
    if not values:
        return "failed"
    return confidence_label(sum(values) / len(values), hard_failed=hard_failed)

