from iona.cv.quality import aggregate_confidence, confidence_gate_issues


def _quality(**overrides):
    quality = {
        "sky_detection": {"confidence": 0.70, "sky_fraction": 0.30},
        "plate_solve": {"success": True, "pixel_scale_arcsec": 170.0},
        "building_lines": {"confidence": 0.90, "line_count": 60, "candidate_vertical_count": 12},
        "vertical_vanishing_point": {
            "success": True,
            "confidence": 0.90,
            "inlier_lines": 12,
            "residual_px": 1.0,
        },
        "camera_model": {"confidence": 0.75, "source": "plate_scale"},
        "rotation_fit": {"success": True, "confidence": 0.95, "residual_deg": 0.2, "sample_count": 80},
        "zenith": {"success": True, "confidence": 1.0, "positive_altitude_fraction": 1.0},
        "time": {"estimated_time_error_seconds": 2.0},
    }
    quality.update(overrides)
    return quality


def test_high_quality_chain_allows_high_confidence() -> None:
    quality = _quality()

    confidence = aggregate_confidence([1.0] * 8, quality=quality)

    assert confidence == "high"
    assert confidence_gate_issues(quality) == []


def test_default_intrinsics_caps_confidence_at_medium() -> None:
    quality = _quality(camera_model={"confidence": 0.35, "source": "default"})

    confidence = aggregate_confidence([1.0] * 8, quality=quality)

    assert confidence == "medium"
    assert confidence_gate_issues(quality)[0]["code"] == "default_intrinsics_used"


def test_weak_vertical_geometry_caps_confidence_at_medium() -> None:
    quality = _quality(
        building_lines={"confidence": 0.55, "line_count": 38, "candidate_vertical_count": 6},
        vertical_vanishing_point={
            "success": True,
            "confidence": 0.89,
            "inlier_lines": 5,
            "residual_px": 0.3,
        },
    )

    confidence = aggregate_confidence([1.0] * 8, quality=quality)

    assert confidence == "medium"
    assert [issue["code"] for issue in confidence_gate_issues(quality)] == ["weak_vertical_geometry"]


def test_weak_zenith_disambiguation_caps_confidence_at_low() -> None:
    quality = _quality(zenith={"success": True, "confidence": 0.45, "positive_altitude_fraction": 0.58})

    confidence = aggregate_confidence([1.0] * 8, quality=quality)

    assert confidence == "low"
    assert confidence_gate_issues(quality)[0]["code"] == "weak_zenith_disambiguation"


def test_solver_failure_keeps_confidence_failed() -> None:
    quality = _quality(plate_solve={"success": False, "failure_reason": "local_solve_field_timeout"})

    confidence = aggregate_confidence([1.0] * 8, hard_failed=True, quality=quality)

    assert confidence == "failed"
