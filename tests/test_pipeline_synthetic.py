from datetime import datetime, timezone

import numpy as np
from PIL import Image

from iona.astronomy.coordinates import radec_to_unit_vector
from iona.astronomy.geolocation import (
    estimate_location_from_zenith,
    zenith_radec_from_location,
)
from iona.camera.intrinsics import estimate_camera_intrinsics
from iona.camera.rotation_fit import fit_rotation_kabsch
from iona.config import PipelineConfig, SolverConfig
from iona.cv.segmentation import SegmentationBackendError
from iona.pipeline import auto_estimate
from iona.pipeline.auto_estimate import estimate_zenith_radec, run_auto_pipeline
from iona.pipeline.result_schema import PlateSolveResult, VanishingPointResult


def test_synthetic_zenith_chain_recovers_known_location() -> None:
    utc = datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
    expected_ra, expected_dec = zenith_radec_from_location(35.72, 139.81, utc)
    zenith_vector = radec_to_unit_vector(expected_ra, expected_dec)
    camera_forward = np.array([0.0, 0.0, 1.0])
    camera_x = np.array([1.0, 0.0, 0.0])
    camera_y = np.array([0.0, 1.0, 0.0])

    celestial_x = np.cross(np.array([0.0, 0.0, 1.0]), zenith_vector)
    if np.linalg.norm(celestial_x) < 1e-6:
        celestial_x = np.array([1.0, 0.0, 0.0])
    celestial_x = celestial_x / np.linalg.norm(celestial_x)
    celestial_y = np.cross(zenith_vector, celestial_x)
    expected_rotation = np.column_stack([celestial_x, celestial_y, zenith_vector])
    rotation = fit_rotation_kabsch(
        np.array([camera_x, camera_y, camera_forward]),
        np.array([celestial_x, celestial_y, zenith_vector]),
    )
    intrinsics = estimate_camera_intrinsics((101, 101))
    vp = VanishingPointResult(
        success=True,
        vanishing_point_homogeneous=np.array([50.0, 50.0, 1.0]),
        inlier_lines=[],
        residual_px=0.0,
        confidence=1.0,
    )
    star_dirs = [zenith_vector, (zenith_vector + celestial_x * 0.05) / np.linalg.norm(zenith_vector + celestial_x * 0.05)]

    zenith = estimate_zenith_radec(vp, intrinsics, rotation, star_dirs)
    location = estimate_location_from_zenith(zenith.ra_deg, zenith.dec_deg, utc)

    assert zenith.success
    assert abs(location.latitude_deg - 35.72) < 1e-9
    assert abs(location.longitude_deg - 139.81) < 1e-9


def test_pipeline_failure_result_contains_machine_readable_diagnostics(tmp_path) -> None:
    image_path = tmp_path / "blank.jpg"
    Image.new("RGB", (80, 60), color=(5, 5, 8)).save(image_path)
    config = PipelineConfig(solver=SolverConfig(solver="none"))

    result = run_auto_pipeline(str(image_path), datetime(2026, 1, 1, tzinfo=timezone.utc), config)

    assert not result.success
    assert "plate_solve_failed" in result.failure_reasons
    assert "not_enough_stars" in result.failure_reasons
    assert "line_detection_failed" in result.failure_reasons
    assert any(event.stage == "plate_solve" and event.status == "failed" for event in result.diagnostics)
    assert any(event.stage == "star_detection" and event.status == "failed" for event in result.diagnostics)
    assert any(event.stage == "line_detection" and event.status == "failed" for event in result.diagnostics)
    assert result.quality["plate_solve"]["failure_reason"] == "plate_solver_disabled"
    assert result.quality["segmentation"]["used_fallback"] is True
    assert any(event.stage == "segmentation" and event.status == "ok" for event in result.diagnostics)


def test_pipeline_uses_exif_transposed_pixels_for_plate_solving(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "portrait_with_orientation.jpg"
    image = Image.new("RGB", (40, 20), color=(5, 5, 8))
    exif = Image.Exif()
    exif[274] = 6
    image.save(image_path, exif=exif)
    observed_sizes = []

    def fake_solve_plate(path, sky_mask, config):
        with Image.open(path) as solver_image:
            observed_sizes.append(solver_image.size)
        return PlateSolveResult(success=False, failure_reason="plate_solver_disabled")

    monkeypatch.setattr(auto_estimate, "solve_plate", fake_solve_plate)

    run_auto_pipeline(
        str(image_path),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        PipelineConfig(solver=SolverConfig(solver="none")),
    )

    assert observed_sizes == [(20, 40)]


def test_pipeline_promotes_solver_timeout_from_attempt_diagnostics(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "blank.jpg"
    Image.new("RGB", (80, 60), color=(5, 5, 8)).save(image_path)

    def fake_solve_plate(path, sky_mask, config):  # noqa: ARG001
        return PlateSolveResult(
            success=False,
            failure_reason="local_solve_field_all_attempts_failed",
            diagnostics={
                "attempt_errors": [
                    {"attempt": "original", "reason": "local_solve_field_no_solution"},
                    {"attempt": "star_enhanced", "reason": "local_solve_field_timeout"},
                ]
            },
        )

    monkeypatch.setattr(auto_estimate, "solve_plate", fake_solve_plate)

    result = run_auto_pipeline(
        str(image_path),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        PipelineConfig(solver=SolverConfig(solver="local")),
    )

    assert "local_solve_field_timeout" in result.failure_reasons
    assert "solver_timeout" in result.failure_reasons


def test_pipeline_reports_explicit_segmentation_backend_failure(tmp_path, monkeypatch) -> None:
    """Regression: explicit SegFormer failures should stay visible in JSON diagnostics."""
    image_path = tmp_path / "blank.jpg"
    Image.new("RGB", (80, 60), color=(5, 5, 8)).save(image_path)

    def fail_segmentation(image, backend, model_id):  # noqa: ARG001
        raise SegmentationBackendError(
            "SegFormer segmentation unavailable",
            backend="segformer",
            model_id="fake/segformer",
            reason="segformer_unavailable",
        )

    monkeypatch.setattr(auto_estimate, "estimate_scene_masks", fail_segmentation)

    result = run_auto_pipeline(
        str(image_path),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        PipelineConfig(
            solver=SolverConfig(solver="none"),
            segmentation_backend="segformer",
            segmentation_model="fake/segformer",
        ),
    )

    assert not result.success
    assert result.confidence == "failed"
    assert "segmentation_failed" in result.failure_reasons
    assert "segformer_unavailable" in result.failure_reasons
    assert result.quality["segmentation"]["failure_reason"] == "segformer_unavailable"
    assert any(event.stage == "segmentation" and event.status == "failed" for event in result.diagnostics)
