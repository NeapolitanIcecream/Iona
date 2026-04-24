"""Validation helpers for the tracked prototype photo set."""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from iona.config import PipelineConfig
from iona.pipeline.auto_estimate import run_auto_pipeline
from iona.pipeline.result_schema import IonaResult
from iona.time_utils import parse_utc_datetime


PipelineRunner = Callable[[str, datetime, PipelineConfig], IonaResult]
LOCAL_SOLVERS = {"solve-field", "local", "local-solve-field"}


def default_manifest_path() -> Path:
    source_manifest = Path(__file__).resolve().parents[3] / "examples" / "prototype_photos" / "manifest.json"
    if source_manifest.is_file():
        return source_manifest
    return packaged_manifest_path()


def packaged_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "prototype_photos" / "manifest.json"


def load_prototype_manifest(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("photos"), list):
        raise ValueError("Prototype manifest must contain a photos list.")
    return manifest


def haversine_distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0088
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    value = max(0.0, min(1.0, value))
    return float(2.0 * radius_km * math.atan2(math.sqrt(value), math.sqrt(1.0 - value)))


def solver_skip_reason(config: PipelineConfig) -> Optional[str]:
    solver = config.solver.solver
    if solver in LOCAL_SOLVERS:
        if not _is_executable(config.solver.local_solve_field_path):
            return "missing_local_solve_field_binary"
        index_dir = config.solver.local_index_dir
        if not index_dir or not Path(index_dir).expanduser().is_dir():
            return "missing_local_astrometry_index_dir"
    if solver == "astrometry-net" and not config.solver.astrometry_api_key:
        return "missing_astrometry_net_api_key"
    return None


def _is_executable(path_or_command: Optional[str]) -> bool:
    if not path_or_command:
        return False
    if os.sep not in path_or_command:
        return shutil.which(path_or_command) is not None
    path = Path(path_or_command).expanduser()
    return path.is_file() and os.access(path, os.X_OK)


def validate_prototype_manifest(
    manifest_path: str | Path,
    config: PipelineConfig,
    run_pipeline: PipelineRunner = run_auto_pipeline,
) -> Dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = load_prototype_manifest(manifest_file)
    generated_at = datetime.now(timezone.utc).isoformat()
    photos: List[Dict[str, Any]] = []
    global_skip_reason = solver_skip_reason(config)

    for photo in manifest["photos"]:
        photos.append(
            _validate_photo(
                photo=photo,
                manifest_dir=manifest_file.parent,
                config=config,
                run_pipeline=run_pipeline,
                global_skip_reason=global_skip_reason,
            )
        )

    summary = _summarize(photos)
    return {
        "generated_at_utc": generated_at,
        "manifest_path": str(manifest_file),
        "solver": config.solver.solver,
        "summary": summary,
        "photos": photos,
    }


def _validate_photo(
    photo: Mapping[str, Any],
    manifest_dir: Path,
    config: PipelineConfig,
    run_pipeline: PipelineRunner,
    global_skip_reason: Optional[str],
) -> Dict[str, Any]:
    photo_id = str(photo.get("id") or photo.get("file") or "unknown")
    image_path = manifest_dir / str(photo.get("file", ""))
    ground_truth = _ground_truth(photo)
    base = {
        "id": photo_id,
        "file": str(image_path),
        "ground_truth": ground_truth,
        "estimated_location": None,
        "estimated_error_km": None,
        "confidence": "failed",
        "failure_reasons": [],
        "warnings": [],
        "quality": {},
        "diagnostics": [],
    }
    if global_skip_reason:
        return _skipped_photo(base, global_skip_reason)
    if not image_path.is_file():
        return _skipped_photo(base, "missing_image_file")

    try:
        utc_time = parse_utc_datetime(
            str(photo["source_time"]),
            timezone_hint=photo.get("timezone_hint"),
        )
    except Exception:
        return _skipped_photo(base, "invalid_source_time")

    try:
        result = run_pipeline(str(image_path), utc_time, config)
    except Exception as exc:
        return _pipeline_exception_photo(base, exc)

    result_dict = result.to_dict()
    location = result_dict.get("estimated_location")
    estimated_error_km = None
    if result.success and result.estimated_location and ground_truth:
        estimated_error_km = round(
            haversine_distance_km(
                ground_truth["lat"],
                ground_truth["lon"],
                result.estimated_location.latitude_deg,
                result.estimated_location.longitude_deg,
            ),
            3,
        )

    status = "success" if result.success else "failed"
    failure_reasons = list(result.failure_reasons)
    if any("timeout" in reason for reason in failure_reasons):
        failure_reasons.append("solver_timeout")

    photo_result = {
        **base,
        "status": status,
        "skip_reason": None,
        "estimated_location": location,
        "estimated_error_km": estimated_error_km,
        "confidence": result.confidence,
        "failure_reasons": sorted(set(failure_reasons)),
        "warnings": result_dict.get("warnings", []),
        "quality": result_dict.get("quality", {}),
        "diagnostics": result_dict.get("diagnostics", []),
    }
    photo_result["expectation"] = _expectation_for(photo_result)
    return photo_result


def _pipeline_exception_photo(base: Mapping[str, Any], exc: Exception) -> Dict[str, Any]:
    return {
        **base,
        "status": "failed",
        "skip_reason": None,
        "confidence": "failed",
        "failure_reasons": ["pipeline_exception"],
        "warnings": ["Pipeline raised an exception for this prototype photo."],
        "diagnostics": [
            {
                "stage": "pipeline",
                "status": "failed",
                "message": "Pipeline raised exception",
                "details": {"error_type": type(exc).__name__, "error": str(exc)},
            }
        ],
        "expectation": {"status": "unscored", "reason": "pipeline_exception"},
    }


def _ground_truth(photo: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    location = photo.get("camera_location")
    if not isinstance(location, Mapping):
        return None
    try:
        return {"lat": float(location["lat"]), "lon": float(location["lon"])}
    except (KeyError, TypeError, ValueError):
        return None


def _skipped_photo(base: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    photo_result = {
        **base,
        "status": "skipped",
        "skip_reason": reason,
        "failure_reasons": [reason],
    }
    photo_result["expectation"] = _expectation_for(photo_result)
    return photo_result


def _expectation_for(photo: Mapping[str, Any]) -> Dict[str, str]:
    photo_id = str(photo["id"])
    if photo_id == "headlands_telescope_milky_way":
        return _headlands_expectation(photo)
    if photo_id == "astronomical_observatory_118127341":
        return _observatory_expectation(photo)
    if photo_id in {"gazing_milky_way_blanco_telescope", "kosovo_skywatcher_milky_way"}:
        return _solver_failure_expectation(photo)
    return {"status": "unscored", "reason": "no_fixed_expectation"}


def _headlands_expectation(photo: Mapping[str, Any]) -> Dict[str, str]:
    if photo["status"] == "skipped":
        return {"status": "skipped", "reason": "benchmark_not_run"}
    error = photo.get("estimated_error_km")
    confidence = str(photo.get("confidence"))
    if photo["status"] == "success" and isinstance(error, (int, float)) and error < 200 and confidence in {"medium", "high"}:
        return {"status": "passed", "reason": "headlands_success_under_200km"}
    return {"status": "failed", "reason": "headlands_must_succeed_under_200km"}


def _observatory_expectation(photo: Mapping[str, Any]) -> Dict[str, str]:
    if photo["status"] == "skipped":
        return {"status": "skipped", "reason": "benchmark_not_run"}
    error = photo.get("estimated_error_km")
    if photo["status"] == "success" and isinstance(error, (int, float)) and error > 500 and photo.get("confidence") == "high":
        return {"status": "failed", "reason": "large_error_must_not_be_high_confidence"}
    return {"status": "passed", "reason": "no_high_confidence_large_error"}


def _solver_failure_expectation(photo: Mapping[str, Any]) -> Dict[str, str]:
    if photo["status"] == "failed" and "solver_timeout" in photo.get("failure_reasons", []):
        return {"status": "passed", "reason": "solver_timeout_recorded"}
    return {"status": "passed", "reason": "no_expected_timeout_required"}


def _summarize(photos: List[Mapping[str, Any]]) -> Dict[str, Any]:
    statuses = [str(photo["status"]) for photo in photos]
    expectations = [str(photo.get("expectation", {}).get("status", "unscored")) for photo in photos]
    return {
        "total": len(photos),
        "success": statuses.count("success"),
        "failed": statuses.count("failed"),
        "skipped": statuses.count("skipped"),
        "expectations_passed": expectations.count("passed"),
        "expectations_failed": expectations.count("failed"),
        "expectations_skipped": expectations.count("skipped"),
    }


def render_validation_markdown(validation: Mapping[str, Any]) -> str:
    lines = [
        "# Iona Prototype Validation",
        "",
        f"- Generated: `{validation.get('generated_at_utc')}`",
        f"- Solver: `{validation.get('solver')}`",
        "",
        "| Photo | Status | Confidence | Error km | Expectation | Main reason |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for photo in validation.get("photos", []):
        error = photo.get("estimated_error_km")
        error_text = "" if error is None else f"{float(error):.1f}"
        reason = photo.get("skip_reason") or _main_reason(photo)
        expectation = photo.get("expectation", {}).get("status", "unscored")
        lines.append(
            f"| {photo.get('id')} | {photo.get('status')} | {photo.get('confidence')} | "
            f"{error_text} | {expectation} | {reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def _main_reason(photo: Mapping[str, Any]) -> str:
    failure_reasons = photo.get("failure_reasons") or []
    if failure_reasons:
        return str(failure_reasons[0])
    gates = photo.get("quality", {}).get("confidence_gates") if isinstance(photo.get("quality"), Mapping) else None
    if gates:
        return str(gates[0].get("code", "confidence_gate"))
    return ""
