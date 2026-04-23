from __future__ import annotations

from io import BytesIO
from pathlib import Path

from iona.config import SolverConfig
from iona.solver import local_solve_field


def _fits_header_bytes() -> bytes:
    from astropy.io import fits

    header = fits.Header()
    header["CRVAL1"] = 12.5
    header["CRVAL2"] = -3.25
    header["CRPIX1"] = 100.0
    header["CRPIX2"] = 50.0
    header["CD1_1"] = -1.0 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    buffer = BytesIO()
    fits.PrimaryHDU(header=header).writeto(buffer)
    return buffer.getvalue()


def test_local_solve_field_returns_wcs_from_solve_field_outputs(tmp_path, monkeypatch) -> None:
    """Local solve should surface WCS and machine-readable command diagnostics."""
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ARG001
        out_dir = Path(cmd[cmd.index("--dir") + 1])
        wcs_path = Path(cmd[cmd.index("--wcs") + 1])
        solved_path = Path(cmd[cmd.index("--solved") + 1])
        wcs_path.write_bytes(_fits_header_bytes())
        solved_path.write_text("1\n")

        class Result:
            returncode = 0
            stdout = "Field 1: solved with index 4110.\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(local_solve_field.subprocess, "run", fake_run)

    result = local_solve_field.solve_with_local_solve_field(
        image_path=str(image_path),
        sky_mask=None,
        config=SolverConfig(
            solver="local",
            timeout_seconds=30,
            local_index_dir=str(index_dir),
            local_solve_field_path="/opt/homebrew/bin/solve-field",
        ),
    )

    assert result.success
    assert result.center_ra_deg == 12.5
    assert result.center_dec_deg == -3.25
    assert result.diagnostics["matched_index"] == "4110"
    assert result.diagnostics["backend"] == "solve-field"
    assert result.diagnostics["index_dir"] == str(index_dir)
    assert "--index-dir" in result.diagnostics["command"]


def test_local_solve_field_reports_missing_index_dir(tmp_path) -> None:
    """Missing index files should fail before spawning solve-field."""
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    result = local_solve_field.solve_with_local_solve_field(
        image_path=str(image_path),
        sky_mask=None,
        config=SolverConfig(
            solver="local",
            timeout_seconds=30,
            local_index_dir=str(tmp_path / "missing-indexes"),
            local_solve_field_path="/opt/homebrew/bin/solve-field",
        ),
    )

    assert not result.success
    assert result.failure_reason == "missing_local_astrometry_index_dir"


def test_local_solve_field_reports_timeout(tmp_path, monkeypatch) -> None:
    """Solver timeout should become a structured PlateSolveResult failure."""
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: ARG001
        raise local_solve_field.subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(local_solve_field.subprocess, "run", fake_run)

    result = local_solve_field.solve_with_local_solve_field(
        image_path=str(image_path),
        sky_mask=None,
        config=SolverConfig(
            solver="local",
            timeout_seconds=7,
            local_index_dir=str(index_dir),
            local_solve_field_path="/opt/homebrew/bin/solve-field",
        ),
    )

    assert not result.success
    assert result.failure_reason == "local_solve_field_timeout"
    assert result.diagnostics["timeout_seconds"] == 7


def test_local_solve_field_parses_index_filename_from_stdout() -> None:
    stdout = "Field 1: solved with index index-4119.fits.\n"

    assert local_solve_field._parse_matched_index(stdout) == "index-4119.fits"
