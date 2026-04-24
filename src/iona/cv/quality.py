"""Rule-based quality aggregation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from iona.pipeline.result_schema import bounded, confidence_label


_CONFIDENCE_RANK = {"failed": 0, "low": 1, "medium": 2, "high": 3}


def _nested_float(quality: Mapping[str, Any], section: str, key: str, default: Optional[float] = None) -> Optional[float]:
    value = quality.get(section, {})
    if not isinstance(value, Mapping):
        return default
    try:
        raw = value.get(key, default)
        return None if raw is None else float(raw)
    except (TypeError, ValueError):
        return default


def _nested_bool(quality: Mapping[str, Any], section: str, key: str, default: bool = False) -> bool:
    value = quality.get(section, {})
    if not isinstance(value, Mapping):
        return default
    return bool(value.get(key, default))


def _nested_str(quality: Mapping[str, Any], section: str, key: str, default: str = "") -> str:
    value = quality.get(section, {})
    if not isinstance(value, Mapping):
        return default
    raw = value.get(key, default)
    return "" if raw is None else str(raw)


def _issue(code: str, max_confidence: str, message: str) -> Dict[str, str]:
    return {"code": code, "max_confidence": max_confidence, "message": message}


def confidence_gate_issues(quality: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Return machine-readable confidence caps implied by stage diagnostics."""

    issues: List[Dict[str, str]] = []
    for gate in (
        _segmentation_gate,
        _plate_solve_gate,
        _sky_gate,
        _vertical_geometry_gate,
        _camera_gate,
        _rotation_gate,
        _zenith_gate,
        _time_gate,
    ):
        issue = gate(quality)
        if issue:
            issues.append(issue)
    return _dedupe_issues(issues)


def _segmentation_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    value = quality.get("segmentation")
    if not isinstance(value, Mapping):
        return None
    confidence = _nested_float(quality, "segmentation", "confidence", 0.0) or 0.0
    sky_fraction = _nested_float(quality, "segmentation", "sky_fraction", 0.0) or 0.0
    building_fraction = _nested_float(quality, "segmentation", "building_fraction", 0.0) or 0.0
    used_fallback = _nested_bool(quality, "segmentation", "used_fallback", default=False)
    if sky_fraction < 0.03 or sky_fraction > 0.92 or building_fraction < 0.005:
        return _issue("implausible_segmentation", "low", "Segmentation mask areas are implausible.")
    if used_fallback:
        return _issue("segmentation_fallback", "medium", "Model segmentation fell back to classic CV masks.")
    if confidence < 0.35:
        return _issue("weak_segmentation", "low", "Segmentation quality is too weak for high confidence.")
    if confidence < 0.55:
        return _issue("weak_segmentation", "medium", "Segmentation quality limits confidence.")
    return None


def _plate_solve_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    if not _nested_bool(quality, "plate_solve", "success", default=False):
        return _issue("plate_solve_failed", "failed", "Plate solving did not produce a usable WCS.")
    return None


def _sky_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    sky_confidence = _nested_float(quality, "sky_detection", "confidence", 0.0) or 0.0
    sky_fraction = _nested_float(quality, "sky_detection", "sky_fraction", 0.0) or 0.0
    if sky_confidence < 0.25 or sky_fraction < 0.08 or sky_fraction > 0.85:
        return _issue("weak_sky_detection", "low", "Sky mask quality is too weak for high confidence.")
    if sky_confidence < 0.45:
        return _issue("weak_sky_detection", "medium", "Sky mask quality limits confidence.")
    return None


def _vertical_geometry_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    building_confidence = _nested_float(quality, "building_lines", "confidence", 0.0) or 0.0
    vertical_count = _nested_float(quality, "building_lines", "candidate_vertical_count", 0.0) or 0.0
    vp_inliers = _nested_float(quality, "vertical_vanishing_point", "inlier_lines", 0.0) or 0.0
    vp_confidence = _nested_float(quality, "vertical_vanishing_point", "confidence", 0.0) or 0.0
    vp_residual = _nested_float(quality, "vertical_vanishing_point", "residual_px", 0.0) or 0.0
    if (
        vertical_count < 8
        or vp_inliers < 8
        or building_confidence < 0.65
        or vp_confidence < 0.75
        or vp_residual > 4.0
    ):
        return _issue(
            "weak_vertical_geometry",
            "medium",
            "Vertical building geometry is too sparse or unstable for high confidence.",
        )
    return None


def _camera_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    camera_source = _nested_str(quality, "camera_model", "source")
    camera_confidence = _nested_float(quality, "camera_model", "confidence", 0.0) or 0.0
    if camera_source == "default" or camera_confidence < 0.5:
        return _issue(
            "default_intrinsics_used",
            "medium",
            "Camera intrinsics came from a low-confidence default estimate.",
        )
    return None


def _rotation_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    rotation_success = _nested_bool(quality, "rotation_fit", "success", default=False)
    rotation_residual = _nested_float(quality, "rotation_fit", "residual_deg", None)
    rotation_confidence = _nested_float(quality, "rotation_fit", "confidence", 0.0) or 0.0
    if not rotation_success:
        return _issue("rotation_fit_failed", "failed", "Camera-to-celestial rotation failed.")
    if rotation_residual is not None and rotation_residual > 3.0:
        return _issue("weak_rotation_fit", "low", "Camera-to-celestial rotation residual is high.")
    if rotation_confidence < 0.85 or (rotation_residual is not None and rotation_residual > 1.0):
        return _issue("weak_rotation_fit", "medium", "Camera-to-celestial rotation limits confidence.")
    return None


def _zenith_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    zenith_success = _nested_bool(quality, "zenith", "success", default=False)
    zenith_confidence = _nested_float(quality, "zenith", "confidence", 0.0) or 0.0
    positive_fraction = _nested_float(quality, "zenith", "positive_altitude_fraction", None)
    if not zenith_success:
        return _issue("zenith_estimation_failed", "failed", "Zenith estimation failed.")
    if positive_fraction is None or zenith_confidence < 0.6 or positive_fraction < 0.65:
        return _issue("weak_zenith_disambiguation", "low", "Zenith/nadir sign disambiguation is weak.")
    if zenith_confidence < 0.75 or positive_fraction < 0.75:
        return _issue("weak_zenith_disambiguation", "medium", "Zenith/nadir sign disambiguation limits confidence.")
    return None


def _time_gate(quality: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    time_error = _nested_float(quality, "time", "estimated_time_error_seconds", 0.0) or 0.0
    if time_error >= 600:
        return _issue("timestamp_uncertain", "low", "Timestamp uncertainty makes longitude unreliable.")
    if time_error >= 60:
        return _issue("timestamp_uncertain", "medium", "Timestamp uncertainty limits longitude confidence.")
    return None


def _dedupe_issues(issues: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: Dict[str, Dict[str, str]] = {}
    for item in issues:
        existing = deduped.get(item["code"])
        if existing is None or _CONFIDENCE_RANK[item["max_confidence"]] < _CONFIDENCE_RANK[existing["max_confidence"]]:
            deduped[item["code"]] = item
    return list(deduped.values())


def apply_confidence_gates(label: str, issues: Iterable[Mapping[str, str]]) -> str:
    gated = label
    for issue in issues:
        cap = str(issue.get("max_confidence", "high"))
        if _CONFIDENCE_RANK.get(cap, 3) < _CONFIDENCE_RANK.get(gated, 0):
            gated = cap
    return gated


def aggregate_confidence(
    scores: Iterable[float],
    hard_failed: bool = False,
    quality: Optional[Mapping[str, Any]] = None,
) -> str:
    values = [bounded(score) for score in scores]
    if not values:
        return "failed"
    label = confidence_label(sum(values) / len(values), hard_failed=hard_failed)
    if quality is None:
        return label
    return apply_confidence_gates(label, confidence_gate_issues(quality))
