from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from iona.config import PipelineConfig, SolverConfig
from iona.pipeline.result_schema import IonaResult, LocationEstimate
from iona.validation.prototypes import (
    _preferred_manifest_path,
    default_manifest_path,
    haversine_distance_km,
    load_prototype_manifest,
    packaged_manifest_path,
    render_validation_markdown,
    validate_prototype_manifest,
)


def _write_manifest(tmp_path, photo_file: str) -> str:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "photos": [
                    {
                        "id": "sample_photo",
                        "file": photo_file,
                        "source_time": "2026-01-01T12:00:00",
                        "timezone_hint": "UTC",
                        "camera_location": {"lat": 35.0, "lon": 139.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(manifest_path)


def test_haversine_distance_reports_degrees_at_equator() -> None:
    distance = haversine_distance_km(0.0, 0.0, 0.0, 1.0)

    assert abs(distance - 111.195) < 0.01


def test_haversine_distance_handles_antipodal_rounding() -> None:
    distance = haversine_distance_km(0.0, 0.0, 0.0, 180.0)

    assert abs(distance - 20015.114) < 0.01


def test_load_prototype_manifest_returns_photo_entries(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path, "sample.jpg")

    manifest = load_prototype_manifest(manifest_path)

    assert manifest["version"] == 1
    assert manifest["photos"][0]["id"] == "sample_photo"


def test_packaged_default_manifest_is_available() -> None:
    packaged_manifest = Path(default_manifest_path())

    assert packaged_manifest.is_file()
    assert load_prototype_manifest(packaged_manifest)["photos"]


def test_packaged_fallback_manifest_references_packaged_images() -> None:
    packaged_manifest = packaged_manifest_path()
    manifest = load_prototype_manifest(packaged_manifest)

    for photo in manifest["photos"]:
        image_path = packaged_manifest.parent / photo["file"]
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.format == "JPEG"


def test_default_manifest_falls_back_when_source_images_are_lfs_pointers(tmp_path) -> None:
    source_dir = tmp_path / "source"
    packaged_dir = tmp_path / "packaged"
    source_dir.mkdir()
    packaged_dir.mkdir()
    source_manifest = Path(_write_manifest(source_dir, "sample.jpg"))
    packaged_manifest = Path(_write_manifest(packaged_dir, "sample.jpg"))
    (source_dir / "sample.jpg").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abcdef\n"
        "size 123\n",
        encoding="utf-8",
    )
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(packaged_dir / "sample.jpg")

    selected_manifest = _preferred_manifest_path(source_manifest, packaged_manifest)

    assert selected_manifest == packaged_manifest


def test_validate_prototype_manifest_skips_when_local_solver_is_unavailable(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    manifest_path = _write_manifest(tmp_path, image_path.name)
    config = PipelineConfig(
        solver=SolverConfig(solver="local", local_solve_field_path=None, local_index_dir=None)
    )

    validation = validate_prototype_manifest(manifest_path, config=config)

    assert validation["summary"]["skipped"] == 1
    assert validation["photos"][0]["status"] == "skipped"
    assert validation["photos"][0]["skip_reason"] == "missing_local_solve_field_binary"


def test_validate_prototype_manifest_skips_when_local_index_dir_is_missing(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    manifest_path = _write_manifest(tmp_path, image_path.name)
    solver_path = tmp_path / "solve-field"
    solver_path.write_text("#!/bin/sh\n", encoding="utf-8")
    solver_path.chmod(0o755)
    config = PipelineConfig(
        solver=SolverConfig(
            solver="local",
            local_solve_field_path=str(solver_path),
            local_index_dir=str(tmp_path / "missing-indexes"),
        )
    )

    validation = validate_prototype_manifest(manifest_path, config=config)

    assert validation["summary"]["skipped"] == 1
    assert validation["photos"][0]["skip_reason"] == "missing_local_astrometry_index_dir"


def test_validate_prototype_manifest_skips_when_solver_path_is_stale(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    manifest_path = _write_manifest(tmp_path, image_path.name)
    config = PipelineConfig(
        solver=SolverConfig(
            solver="local",
            local_solve_field_path=str(tmp_path / "missing-solve-field"),
            local_index_dir=str(tmp_path),
        )
    )

    validation = validate_prototype_manifest(manifest_path, config=config)

    assert validation["summary"]["skipped"] == 1
    assert validation["photos"][0]["skip_reason"] == "missing_local_solve_field_binary"


def test_skipped_observatory_expectation_counts_as_skipped(tmp_path) -> None:
    image_path = tmp_path / "observatory.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "photos": [
                    {
                        "id": "astronomical_observatory_118127341",
                        "file": image_path.name,
                        "source_time": "2026-01-01T12:00:00",
                        "timezone_hint": "UTC",
                        "camera_location": {"lat": 42.4463, "lon": 13.5604},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    validation = validate_prototype_manifest(
        manifest_path,
        config=PipelineConfig(
            solver=SolverConfig(solver="local", local_solve_field_path=None, local_index_dir=None)
        ),
    )

    assert validation["photos"][0]["status"] == "skipped"
    assert validation["photos"][0]["expectation"]["status"] == "skipped"
    assert validation["summary"]["expectations_skipped"] == 1
    assert validation["summary"]["expectations_passed"] == 0


def test_skipped_solver_failure_expectation_counts_as_skipped(tmp_path) -> None:
    image_path = tmp_path / "blanco.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "photos": [
                    {
                        "id": "gazing_milky_way_blanco_telescope",
                        "file": image_path.name,
                        "source_time": "2026-01-01T12:00:00",
                        "timezone_hint": "UTC",
                        "camera_location": {"lat": -30.1691, "lon": -70.8046},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    validation = validate_prototype_manifest(
        manifest_path,
        config=PipelineConfig(
            solver=SolverConfig(solver="local", local_solve_field_path=None, local_index_dir=None)
        ),
    )

    assert validation["photos"][0]["status"] == "skipped"
    assert validation["photos"][0]["expectation"]["status"] == "skipped"
    assert validation["summary"]["expectations_skipped"] == 1
    assert validation["summary"]["expectations_passed"] == 0


def test_validate_prototype_manifest_computes_benchmark_error_with_fake_runner(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    manifest_path = _write_manifest(tmp_path, image_path.name)

    def fake_runner(image_path: str, utc_time: datetime, config: PipelineConfig) -> IonaResult:
        assert image_path.endswith("sample.jpg")
        assert utc_time == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        return IonaResult(
            success=True,
            estimated_location=LocationEstimate(latitude_deg=35.0, longitude_deg=139.0, gmst_deg=0.0),
            confidence="medium",
            quality={"plate_solve": {"success": True}},
            warnings=[],
            failure_reasons=[],
            diagnostics=[],
        )

    validation = validate_prototype_manifest(
        manifest_path,
        config=PipelineConfig(solver=SolverConfig(solver="none")),
        run_pipeline=fake_runner,
    )

    assert validation["summary"]["success"] == 1
    assert validation["photos"][0]["estimated_error_km"] == 0.0


def test_validate_prototype_manifest_records_pipeline_exceptions_and_continues(tmp_path) -> None:
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(first_image)
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(second_image)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "photos": [
                    {
                        "id": "first",
                        "file": first_image.name,
                        "source_time": "2026-01-01T12:00:00",
                        "timezone_hint": "UTC",
                        "camera_location": {"lat": 35.0, "lon": 139.0},
                    },
                    {
                        "id": "second",
                        "file": second_image.name,
                        "source_time": "2026-01-01T12:00:00",
                        "timezone_hint": "UTC",
                        "camera_location": {"lat": 35.0, "lon": 139.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(image_path: str, utc_time: datetime, config: PipelineConfig) -> IonaResult:
        if image_path.endswith("first.jpg"):
            raise RuntimeError("corrupt image")
        return IonaResult(
            success=True,
            estimated_location=LocationEstimate(latitude_deg=35.0, longitude_deg=139.0, gmst_deg=0.0),
            confidence="medium",
            quality={},
            warnings=[],
            failure_reasons=[],
            diagnostics=[],
        )

    validation = validate_prototype_manifest(
        manifest_path,
        config=PipelineConfig(solver=SolverConfig(solver="none")),
        run_pipeline=fake_runner,
    )

    assert validation["summary"]["failed"] == 1
    assert validation["summary"]["success"] == 1
    assert validation["photos"][0]["failure_reasons"] == ["pipeline_exception"]
    assert validation["photos"][0]["diagnostics"][0]["details"]["error_type"] == "RuntimeError"


def test_render_validation_markdown_includes_skipped_diagnostics(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(image_path)
    validation = validate_prototype_manifest(
        _write_manifest(tmp_path, image_path.name),
        config=PipelineConfig(
            solver=SolverConfig(solver="local", local_solve_field_path=None, local_index_dir=None)
        ),
    )

    markdown = render_validation_markdown(validation)

    assert "| sample_photo | skipped |" in markdown
    assert "missing_local_solve_field_binary" in markdown
