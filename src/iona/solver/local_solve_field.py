"""Local astrometry.net solve-field integration."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from iona.pipeline.result_schema import PlateSolveResult


def solve_with_local_solve_field(image_path: str, sky_mask: Optional[np.ndarray], config: Any) -> PlateSolveResult:
    solve_field_path = getattr(config, "local_solve_field_path", None)
    if not solve_field_path:
        return PlateSolveResult(
            success=False,
            failure_reason="missing_local_solve_field_binary",
            diagnostics={"backend": "solve-field"},
        )

    index_dir = getattr(config, "local_index_dir", None)
    if not index_dir:
        return PlateSolveResult(
            success=False,
            failure_reason="missing_local_astrometry_index_dir",
            diagnostics={"backend": "solve-field"},
        )

    index_path = Path(index_dir).expanduser()
    if not index_path.is_dir():
        return PlateSolveResult(
            success=False,
            failure_reason="missing_local_astrometry_index_dir",
            diagnostics={"backend": "solve-field", "index_dir": str(index_path)},
        )

    with tempfile.TemporaryDirectory(prefix="iona-solve-field-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        wcs_path = tmp_path / "solve.wcs"
        solved_path = tmp_path / "solve.solved"
        corr_path = tmp_path / "solve.corr"
        rdls_path = tmp_path / "solve.rdls"
        match_path = tmp_path / "solve.match"

        command = [
            solve_field_path,
            image_path,
            "--overwrite",
            "--no-plots",
            "--dir",
            str(tmp_path),
            "--out",
            "solve",
            "--wcs",
            str(wcs_path),
            "--solved",
            str(solved_path),
            "--corr",
            str(corr_path),
            "--rdls",
            str(rdls_path),
            "--match",
            str(match_path),
            "--new-fits",
            "none",
            "--index-dir",
            str(index_path),
            "--cpulimit",
            str(int(getattr(config, "timeout_seconds", 600))),
            "--downsample",
            str(max(1, int(getattr(config, "local_downsample", 2)))),
        ]

        backend_config = getattr(config, "local_backend_config", None)
        if backend_config:
            command.extend(["--backend-config", str(Path(backend_config).expanduser())])

        scale_low = getattr(config, "local_scale_low", None)
        scale_high = getattr(config, "local_scale_high", None)
        if scale_low is not None and scale_high is not None:
            command.extend(
                [
                    "--scale-low",
                    str(scale_low),
                    "--scale-high",
                    str(scale_high),
                    "--scale-units",
                    str(getattr(config, "local_scale_units", "degwidth")),
                ]
            )

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=int(getattr(config, "timeout_seconds", 600)) + 30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PlateSolveResult(
                success=False,
                failure_reason="local_solve_field_timeout",
                diagnostics={
                    "backend": "solve-field",
                    "command": command,
                    "index_dir": str(index_path),
                    "timeout_seconds": int(getattr(config, "timeout_seconds", 600)),
                },
            )

        diagnostics: Dict[str, Any] = {
            "backend": "solve-field",
            "command": command,
            "index_dir": str(index_path),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "solved_file_exists": solved_path.exists(),
            "wcs_file_exists": wcs_path.exists(),
        }

        if not wcs_path.exists() or wcs_path.stat().st_size == 0:
            return PlateSolveResult(
                success=False,
                failure_reason="local_solve_field_no_solution",
                diagnostics=diagnostics,
            )

        try:
            header = _read_wcs_header(wcs_path)
        except Exception as exc:
            diagnostics["wcs_error"] = str(exc)
            return PlateSolveResult(
                success=False,
                failure_reason="invalid_wcs_header",
                diagnostics=diagnostics,
            )

        diagnostics["matched_index"] = _parse_matched_index(proc.stdout)
        return PlateSolveResult(
            success=True,
            wcs_header=header,
            center_ra_deg=_float_or_none(header.get("CRVAL1")),
            center_dec_deg=_float_or_none(header.get("CRVAL2")),
            pixel_scale_arcsec=_pixel_scale_arcsec(header),
            orientation_deg=_orientation_deg(header),
            residual_arcsec=None,
            raw={
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-8000:],
            },
            diagnostics=diagnostics,
        )


def _read_wcs_header(wcs_path: Path) -> Dict[str, Any]:
    from astropy.io import fits

    with fits.open(wcs_path) as hdul:
        return dict(hdul[0].header)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _pixel_scale_arcsec(header: Dict[str, Any]) -> Optional[float]:
    try:
        from astropy.io.fits import Header
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
    except Exception:
        return None

    filtered = Header()
    for key, value in header.items():
        keyword = str(key).strip()
        if keyword.upper() in {"", "COMMENT", "HISTORY", "END"}:
            continue
        try:
            filtered[keyword] = value
        except Exception:
            continue

    try:
        scales = proj_plane_pixel_scales(WCS(filtered)) * 3600.0
        if len(scales) == 0:
            return None
        return float(np.mean(scales))
    except Exception:
        return None


def _orientation_deg(header: Dict[str, Any]) -> Optional[float]:
    cd11 = _float_or_none(header.get("CD1_1"))
    cd21 = _float_or_none(header.get("CD2_1"))
    if cd11 is None or cd21 is None:
        return None
    return float(np.degrees(np.arctan2(cd21, cd11)))


def _parse_matched_index(stdout: str) -> Optional[str]:
    match = re.search(r"solved with index\s+([^\s.]+(?:\.fits)?)", stdout, re.IGNORECASE)
    return match.group(1) if match else None
