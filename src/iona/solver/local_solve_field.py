"""Local astrometry.net solve-field integration."""

from __future__ import annotations

import re
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from iona.pipeline.result_schema import PlateSolveResult
from iona.solver.image_variants import (
    cleanup_solver_image_variants,
    make_solver_image_variants,
)


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

    variants = make_solver_image_variants(image_path, sky_mask)
    try:
        return _solve_local_variants(variants, config, index_path)
    finally:
        cleanup_solver_image_variants(variants)


def _solve_local_variants(variants: list[Any], config: Any, index_path: Path) -> PlateSolveResult:
    timeout_seconds = int(getattr(config, "timeout_seconds", 600))
    deadline = time.monotonic() + timeout_seconds
    attempt_errors = []
    last_result: Optional[PlateSolveResult] = None

    for attempt_index, variant in enumerate(variants):
        remaining_seconds = _remaining_attempt_seconds(attempt_index, timeout_seconds, deadline)
        result = _solve_local_variant_or_timeout(variant.path, config, index_path, remaining_seconds, timeout_seconds)
        last_result = result
        result.diagnostics["attempt_label"] = variant.label
        if result.success:
            result.diagnostics["attempt_errors"] = attempt_errors
            return result
        attempt_errors.append(_attempt_error(variant.label, result))

    if len(variants) == 1 and last_result is not None:
        return last_result
    return PlateSolveResult(
        success=False,
        failure_reason=_collapsed_failure_reason(attempt_errors),
        diagnostics={
            "backend": "solve-field",
            "index_dir": str(index_path),
            "timeout_seconds": timeout_seconds,
            "attempt_errors": attempt_errors,
        },
    )


def _remaining_attempt_seconds(attempt_index: int, timeout_seconds: int, deadline: float) -> float:
    return float(timeout_seconds) if attempt_index == 0 else deadline - time.monotonic()


def _solve_local_variant_or_timeout(
    image_path: str,
    config: Any,
    index_path: Path,
    remaining_seconds: float,
    timeout_seconds: int,
) -> PlateSolveResult:
    if remaining_seconds <= 0:
        return PlateSolveResult(
            success=False,
            failure_reason="local_solve_field_timeout",
            diagnostics={
                "backend": "solve-field",
                "index_dir": str(index_path),
                "timeout_seconds": timeout_seconds,
            },
        )
    return _solve_single_local_variant(
        image_path,
        config,
        index_path,
        remaining_timeout_seconds=remaining_seconds,
        total_timeout_seconds=timeout_seconds,
    )


def _attempt_error(label: str, result: PlateSolveResult) -> Dict[str, Any]:
    return {
        "attempt": label,
        "reason": result.failure_reason or "unknown",
        "returncode": result.diagnostics.get("returncode"),
        "stdout_tail": result.diagnostics.get("stdout_tail"),
        "stderr_tail": result.diagnostics.get("stderr_tail"),
        "matched_index": result.diagnostics.get("matched_index"),
    }


def _solve_single_local_variant(
    image_path: str,
    config: Any,
    index_path: Path,
    remaining_timeout_seconds: Optional[float] = None,
    total_timeout_seconds: Optional[int] = None,
) -> PlateSolveResult:
    solve_field_path = getattr(config, "local_solve_field_path", None)
    configured_timeout = int(getattr(config, "timeout_seconds", 600))
    solver_timeout_seconds = max(
        1,
        math.ceil(remaining_timeout_seconds if remaining_timeout_seconds is not None else configured_timeout),
    )
    total_timeout_seconds = configured_timeout if total_timeout_seconds is None else total_timeout_seconds
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
            str(solver_timeout_seconds),
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
                timeout=solver_timeout_seconds,
                check=False,
            )
        except OSError as exc:
            return PlateSolveResult(
                success=False,
                failure_reason="local_solve_field_launch_failed",
                diagnostics={
                    "backend": "solve-field",
                    "command": command,
                    "index_dir": str(index_path),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        except subprocess.TimeoutExpired:
            return PlateSolveResult(
                success=False,
                failure_reason="local_solve_field_timeout",
                diagnostics={
                    "backend": "solve-field",
                    "command": command,
                    "index_dir": str(index_path),
                    "timeout_seconds": total_timeout_seconds,
                    "attempt_timeout_seconds": solver_timeout_seconds,
                },
            )

        diagnostics: Dict[str, Any] = {
            "backend": "solve-field",
            "command": command,
            "index_dir": str(index_path),
            "timeout_seconds": total_timeout_seconds,
            "attempt_timeout_seconds": solver_timeout_seconds,
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


def _collapsed_failure_reason(attempt_errors: list[Dict[str, Any]]) -> str:
    reasons = [
        str(error.get("reason"))
        for error in attempt_errors
        if isinstance(error, dict) and error.get("reason")
    ]
    if "local_solve_field_timeout" in reasons:
        return "local_solve_field_timeout"
    if reasons and all(reason == reasons[0] for reason in reasons) and reasons[0] != "local_solve_field_no_solution":
        return reasons[0]
    return "local_solve_field_all_attempts_failed"


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
