"""End-to-end automatic Iona pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from iona.astronomy.coordinates import radec_to_unit_vector, unit_vector_to_radec
from iona.astronomy.geolocation import estimate_location_from_zenith
from iona.camera.intrinsics import estimate_camera_intrinsics
from iona.camera.rays import image_point_to_camera_ray
from iona.camera.rotation_fit import fit_camera_to_celestial_rotation
from iona.config import PipelineConfig
from iona.cv.line_detection import detect_building_lines
from iona.cv.preprocess import load_rgb_image, save_rgb_image_temp
from iona.cv.quality import aggregate_confidence
from iona.cv.sky_mask import estimate_sky_mask
from iona.cv.star_detection import detect_star_candidates
from iona.cv.vanishing_point import estimate_vertical_vanishing_point
from iona.exif import read_exif
from iona.pipeline.result_schema import (
    IonaResult,
    PipelineEvent,
    PlateSolveResult,
    RotationFitResult,
    ZenithEstimate,
)
from iona.solver.astrometry_net import solve_plate


def _event(stage: str, status: str, message: str, **details: Any) -> PipelineEvent:
    return PipelineEvent(stage=stage, status=status, message=message, details=details)


def _sample_sky_pixels(mask: Optional[np.ndarray], width: int, height: int, limit: int = 80) -> List[Tuple[float, float]]:
    if mask is None or mask.shape[:2] != (height, width) or not np.any(mask):
        xs = np.linspace(width * 0.25, width * 0.75, 5)
        ys = np.linspace(height * 0.20, height * 0.70, 5)
        return [(float(x), float(y)) for y in ys for x in xs]
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    step = max(1, len(xs) // limit)
    return [(float(x), float(y)) for x, y in zip(xs[::step][:limit], ys[::step][:limit])]


def _star_dirs_from_wcs(wcs: object, points: Sequence[Any], max_points: int = 80) -> List[np.ndarray]:
    if wcs is None or not points:
        return []
    selected = list(points[:max_points])
    try:
        xs = np.array([point.x for point in selected], dtype=float)
        ys = np.array([point.y for point in selected], dtype=float)
        coords = wcs.pixel_to_world(xs, ys)
        return [
            radec_to_unit_vector(float(ra), float(dec))
            for ra, dec in zip(coords.ra.degree, coords.dec.degree)
            if np.isfinite(ra) and np.isfinite(dec)
        ]
    except Exception:
        return []


def estimate_zenith_radec(
    vanishing_point: Any,
    intrinsics: Any,
    rotation: RotationFitResult,
    solved_star_dirs: Sequence[np.ndarray],
) -> ZenithEstimate:
    if not vanishing_point.success or vanishing_point.vanishing_point_homogeneous is None:
        return ZenithEstimate(
            success=False,
            ra_deg=None,
            dec_deg=None,
            zenith_vector=None,
            selected_sign=None,
            positive_altitude_fraction=None,
            confidence=0.0,
            failure_reason="missing_vanishing_point",
        )
    if not rotation.success or rotation.rotation_matrix is None:
        return ZenithEstimate(
            success=False,
            ra_deg=None,
            dec_deg=None,
            zenith_vector=None,
            selected_sign=None,
            positive_altitude_fraction=None,
            confidence=0.0,
            failure_reason="missing_camera_to_celestial_rotation",
        )

    camera_ray = image_point_to_camera_ray(vanishing_point.vanishing_point_homogeneous, intrinsics)
    celestial_ray = rotation.rotation_matrix @ camera_ray
    candidates = [(1, celestial_ray), (-1, -celestial_ray)]

    if solved_star_dirs:
        scored = []
        for sign, candidate in candidates:
            dots = [float(np.dot(star_dir, candidate)) for star_dir in solved_star_dirs]
            fraction = sum(dot > 0 for dot in dots) / len(dots)
            margin = float(np.median(dots)) if dots else 0.0
            scored.append((fraction, margin, sign, candidate))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        fraction, margin, sign, zenith_vector = scored[0]
        confidence = max(0.0, min(1.0, 0.5 + abs(scored[0][0] - scored[1][0])))
        warnings: List[str] = []
        if fraction < 0.55:
            warnings.append("Zenith/nadir sign disambiguation is weak.")
            confidence *= 0.6
    else:
        sign, zenith_vector = candidates[0]
        fraction = None
        margin = None
        confidence = 0.35
        warnings = ["No solved star directions available for zenith/nadir disambiguation."]

    ra_deg, dec_deg = unit_vector_to_radec(zenith_vector)
    return ZenithEstimate(
        success=True,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        zenith_vector=zenith_vector,
        selected_sign=sign,
        positive_altitude_fraction=fraction,
        confidence=float(confidence),
        warnings=warnings,
        diagnostics={"median_star_dot": margin},
    )


def _quality_dict(**items: Any) -> Dict[str, Any]:
    return {key: value for key, value in items.items() if value is not None}


def run_auto_pipeline(
    image_path: str,
    utc_time: datetime,
    config: PipelineConfig,
) -> IonaResult:
    diagnostics: List[PipelineEvent] = []
    warnings: List[str] = []
    failure_reasons: List[str] = []

    exif_info = read_exif(image_path)
    if exif_info.gps_present_ignored:
        warnings.append("EXIF GPS tags were present and ignored.")
    diagnostics.append(_event("exif", "ok", "EXIF read completed", gps_present_ignored=exif_info.gps_present_ignored))

    image = load_rgb_image(image_path)
    height, width = image.shape[:2]
    diagnostics.append(_event("image", "ok", "Image loaded", width=width, height=height))

    sky = estimate_sky_mask(image)
    warnings.extend(sky.warnings)
    diagnostics.append(_event("sky_detection", "ok", "Sky mask estimated", **sky.diagnostics))

    stars = detect_star_candidates(image, sky.sky_mask) if sky.sky_mask is not None else None
    if stars is None:
        failure_reasons.append("star_detection_failed")
        diagnostics.append(_event("star_detection", "failed", "Star detection could not run"))
    else:
        warnings.extend(stars.warnings)
        star_details = {**stars.diagnostics, "min_star_count": config.min_star_count}
        if stars.star_count < config.min_star_count:
            failure_reasons.append("not_enough_stars")
            warnings.append("Too few star candidates detected for reliable zenith/nadir disambiguation.")
            diagnostics.append(
                _event("star_detection", "failed", "Too few star candidates detected", **star_details)
            )
        else:
            diagnostics.append(_event("star_detection", "ok", "Star candidates detected", **star_details))

    lines = detect_building_lines(image, sky.sky_mask) if sky.sky_mask is not None else None
    if lines:
        warnings.extend(lines.warnings)
        diagnostics.append(_event("line_detection", "ok", "Building lines detected", **lines.diagnostics))
    else:
        failure_reasons.append("line_detection_failed")
        diagnostics.append(_event("line_detection", "failed", "Building line detection could not run"))

    solver_image_path = save_rgb_image_temp(image)
    try:
        plate: PlateSolveResult = solve_plate(solver_image_path, sky.sky_mask, config.solver)
    finally:
        try:
            Path(solver_image_path).unlink()
        except FileNotFoundError:
            pass

    if not plate.success:
        failure_reasons.append("plate_solve_failed")
        if plate.failure_reason:
            failure_reasons.append(plate.failure_reason)
        diagnostics.append(
            _event(
                "plate_solve",
                "failed",
                "Plate solving failed",
                input_orientation="exif_transposed",
                **plate.diagnostics,
            )
        )
    else:
        diagnostics.append(
            _event(
                "plate_solve",
                "ok",
                "Plate solving succeeded",
                input_orientation="exif_transposed",
                **plate.diagnostics,
            )
        )

    intrinsics = estimate_camera_intrinsics((height, width), exif_info=exif_info, plate_result=plate)
    warnings.extend(intrinsics.warnings)
    diagnostics.append(_event("intrinsics", "ok", "Camera intrinsics estimated", source=intrinsics.source))

    vp = None
    if lines:
        vp = estimate_vertical_vanishing_point(
            lines.candidate_vertical_lines,
            (height, width),
            min_inliers=config.min_vertical_lines,
        )
        warnings.extend(vp.warnings)
        if vp.success:
            diagnostics.append(_event("vanishing_point", "ok", "Vertical vanishing point estimated", **vp.diagnostics))
        else:
            failure_reasons.append("vanishing_point_failed")
            if vp.failure_reason:
                failure_reasons.append(vp.failure_reason)
            diagnostics.append(_event("vanishing_point", "failed", "Vertical vanishing point failed", **vp.diagnostics))

    wcs = plate.to_wcs() if plate.success else None
    sample_pixels = _sample_sky_pixels(sky.sky_mask, width, height)
    rotation = fit_camera_to_celestial_rotation(wcs, intrinsics, sample_pixels)
    if not rotation.success:
        failure_reasons.append("rotation_fit_failed")
        if rotation.failure_reason:
            failure_reasons.append(rotation.failure_reason)
        diagnostics.append(_event("rotation_fit", "failed", "Camera-to-celestial rotation failed", **rotation.diagnostics))
    else:
        diagnostics.append(_event("rotation_fit", "ok", "Camera-to-celestial rotation fitted", **rotation.diagnostics))

    usable_star_points = (
        stars.star_candidates if stars is not None and stars.star_count >= config.min_star_count else []
    )
    solved_star_dirs = _star_dirs_from_wcs(wcs, usable_star_points)
    zenith = estimate_zenith_radec(vp, intrinsics, rotation, solved_star_dirs) if vp else None
    if zenith and zenith.success:
        warnings.extend(zenith.warnings)
        diagnostics.append(_event("zenith", "ok", "Zenith RA/Dec estimated", **zenith.diagnostics))
        location = estimate_location_from_zenith(
            zenith.ra_deg,
            zenith.dec_deg,
            utc_time,
            estimated_time_error_seconds=config.time_error_seconds,
        )
    else:
        location = None
        failure_reasons.append("zenith_estimation_failed")
        if zenith and zenith.failure_reason:
            failure_reasons.append(zenith.failure_reason)
        diagnostics.append(_event("zenith", "failed", "Zenith estimation failed"))

    quality = {
        "sky_detection": _quality_dict(confidence=sky.confidence, sky_fraction=sky.sky_fraction),
        "star_detection": _quality_dict(
            confidence=stars.confidence if stars else 0.0,
            star_count=stars.star_count if stars else 0,
            star_density=stars.star_density if stars else 0,
            min_star_count=config.min_star_count,
        ),
        "plate_solve": _quality_dict(
            success=plate.success,
            matched_stars=plate.matched_stars,
            pixel_scale_arcsec=plate.pixel_scale_arcsec,
            residual_arcsec=plate.residual_arcsec,
            failure_reason=plate.failure_reason,
        ),
        "building_lines": _quality_dict(
            confidence=lines.confidence if lines else 0.0,
            line_count=len(lines.line_segments) if lines else 0,
            candidate_vertical_count=len(lines.candidate_vertical_lines) if lines else 0,
        ),
        "vertical_vanishing_point": _quality_dict(
            success=vp.success if vp else False,
            confidence=vp.confidence if vp else 0.0,
            inlier_lines=len(vp.inlier_lines) if vp else 0,
            residual_px=vp.residual_px if vp else None,
            failure_reason=vp.failure_reason if vp else None,
        ),
        "camera_model": _quality_dict(confidence=intrinsics.confidence, source=intrinsics.source),
        "rotation_fit": _quality_dict(
            success=rotation.success,
            confidence=rotation.confidence,
            residual_deg=rotation.residual_deg,
            sample_count=rotation.sample_count,
            failure_reason=rotation.failure_reason,
        ),
        "zenith": _quality_dict(
            success=zenith.success if zenith else False,
            confidence=zenith.confidence if zenith else 0.0,
            positive_altitude_fraction=zenith.positive_altitude_fraction if zenith else None,
            failure_reason=zenith.failure_reason if zenith else None,
        ),
        "time": _quality_dict(
            utc=utc_time.isoformat(),
            estimated_time_error_seconds=config.time_error_seconds,
            estimated_lon_error_deg=config.time_error_seconds * 15.0 / 3600.0,
        ),
    }
    hard_failed = location is None or bool(failure_reasons)
    confidence = aggregate_confidence(
        [
            sky.confidence,
            stars.confidence if stars else 0.0,
            1.0 if plate.success else 0.0,
            lines.confidence if lines else 0.0,
            vp.confidence if vp else 0.0,
            intrinsics.confidence,
            rotation.confidence,
            zenith.confidence if zenith else 0.0,
        ],
        hard_failed=hard_failed,
    )
    return IonaResult(
        success=location is not None and not failure_reasons,
        estimated_location=location,
        confidence=confidence,
        quality=quality,
        warnings=sorted(set(warnings)),
        failure_reasons=sorted(set(failure_reasons)),
        diagnostics=diagnostics,
    )
