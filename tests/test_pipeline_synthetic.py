from datetime import datetime, timezone

import numpy as np
from PIL import Image

from astrogeo.astronomy.coordinates import radec_to_unit_vector
from astrogeo.astronomy.geolocation import (
    estimate_location_from_zenith,
    zenith_radec_from_location,
)
from astrogeo.camera.intrinsics import estimate_camera_intrinsics
from astrogeo.camera.rotation_fit import fit_rotation_kabsch
from astrogeo.pipeline.auto_estimate import estimate_zenith_radec, run_auto_pipeline
from astrogeo.config import PipelineConfig, SolverConfig
from astrogeo.pipeline.result_schema import VanishingPointResult


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
    assert any(event.stage == "plate_solve" and event.status == "failed" for event in result.diagnostics)
    assert result.quality["plate_solve"]["failure_reason"] == "plate_solver_disabled"

