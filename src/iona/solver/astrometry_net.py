"""Astrometry.net API wrapper.

The wrapper uploads one or more automatic image variants and returns a WCS-backed
plate-solve result when the service succeeds.
"""

from __future__ import annotations

import json
import re
import time
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from iona.config import SolverConfig
from iona.pipeline.result_schema import PlateSolveResult
from iona.solver.local_solve_field import solve_with_local_solve_field


API_BASE = "https://nova.astrometry.net/api"
PUBLIC_BASE = "https://nova.astrometry.net"


class AstrometryNetClient:
    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        request_max_attempts: int = 4,
        request_retry_delay_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.session_key: Optional[str] = None
        self.request_max_attempts = max(1, request_max_attempts)
        self.request_retry_delay_seconds = max(0.0, request_retry_delay_seconds)
        self.retry_events: List[Dict[str, Any]] = []

    def _retry_delay(self, attempt: int) -> float:
        return self.request_retry_delay_seconds * (2 ** max(0, attempt - 1))

    def _record_retry(self, operation: str, attempt: int, reason: str) -> None:
        self.retry_events.append({"operation": operation, "attempt": attempt, "reason": reason})

    def _request_with_retries(self, operation: str, send_request: Any) -> requests.Response:
        retry_statuses = {429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.request_max_attempts + 1):
            try:
                response = send_request()
                status_code = getattr(response, "status_code", None)
                if status_code in retry_statuses and attempt < self.request_max_attempts:
                    self._record_retry(operation, attempt, f"http_{status_code}")
                    time.sleep(self._retry_delay(attempt))
                    continue
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in retry_statuses and attempt < self.request_max_attempts:
                    last_error = exc
                    self._record_retry(operation, attempt, f"http_{status_code}")
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise
            except requests.RequestException as exc:
                if attempt >= self.request_max_attempts:
                    raise
                last_error = exc
                self._record_retry(operation, attempt, type(exc).__name__)
                time.sleep(self._retry_delay(attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Astrometry.net request failed without a response: {operation}")

    def login(self) -> str:
        response = self._request_with_retries(
            "login",
            lambda: self.session.post(
                f"{API_BASE}/login",
                data={"request-json": json.dumps({"apikey": self.api_key})},
                timeout=30,
            ),
        )
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("errormessage") or "Astrometry.net login failed.")
        self.session_key = payload["session"]
        return self.session_key

    def upload(self, image_path: str) -> int:
        if not self.session_key:
            self.login()
        request = {
            "session": self.session_key,
            "allow_commercial_use": "n",
            "allow_modifications": "n",
            "publicly_visible": "n",
        }

        def send_upload() -> requests.Response:
            with open(image_path, "rb") as handle:
                return self.session.post(
                    f"{API_BASE}/upload",
                    data={"request-json": json.dumps(request)},
                    files={"file": handle},
                    timeout=120,
                )

        response = self._request_with_retries("upload", send_upload)
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("errormessage") or "Astrometry.net upload failed.")
        return int(payload["subid"])

    def wait_for_job(self, submission_id: int, timeout_seconds: int, poll_interval_seconds: float) -> int:
        deadline = time.monotonic() + timeout_seconds
        last_payload: Dict[str, Any] = {}
        job_id: Optional[int] = None
        last_info: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            if job_id is None:
                response = self._request_with_retries(
                    "submission_status",
                    lambda: self.session.get(f"{API_BASE}/submissions/{submission_id}", timeout=30),
                )
                payload = response.json()
                last_payload = payload
                jobs = [job for job in payload.get("jobs", []) if job]
                if jobs:
                    job_id = int(jobs[0])

            if job_id is not None:
                last_info = self.job_info(job_id)
                status = str(last_info.get("status") or "").lower()
                if status == "success":
                    return job_id
                if status in {"failure", "failed", "error"}:
                    raise RuntimeError(f"Astrometry.net job failed: {last_info}")

            time.sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Astrometry.net submission timed out: submission={last_payload}, job={last_info}"
        )

    def job_info(self, job_id: int) -> Dict[str, Any]:
        response = self._request_with_retries(
            "job_info",
            lambda: self.session.get(f"{API_BASE}/jobs/{job_id}/info", timeout=30),
        )
        return response.json()

    def wcs_header(self, job_id: int) -> Dict[str, Any]:
        response = self._request_with_retries(
            "wcs_header",
            lambda: self.session.get(f"{PUBLIC_BASE}/wcs_file/{job_id}", timeout=60),
        )
        try:
            from astropy.io import fits

            with fits.open(BytesIO(response.content)) as hdul:
                return dict(hdul[0].header)
        except Exception as exc:
            raise RuntimeError("Astrometry.net WCS download was not a valid FITS header.") from exc

    def solve(self, image_path: str, config: SolverConfig) -> PlateSolveResult:
        submission_id = self.upload(image_path)
        job_id = self.wait_for_job(
            submission_id,
            timeout_seconds=config.timeout_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        info = self.job_info(job_id)
        if info.get("status") not in (None, "success"):
            return PlateSolveResult(
                success=False,
                failure_reason="astrometry_job_failed",
                raw=info,
                diagnostics={
                    "submission_id": submission_id,
                    "job_id": job_id,
                    "retry_events": list(self.retry_events),
                },
            )
        calibration = info.get("calibration") or {}
        try:
            header = self.wcs_header(job_id)
        except Exception as exc:
            return PlateSolveResult(
                success=False,
                failure_reason="invalid_wcs_header",
                raw=info,
                diagnostics={
                    "submission_id": submission_id,
                    "job_id": job_id,
                    "error": str(exc),
                    "retry_events": list(self.retry_events),
                },
            )
        return PlateSolveResult(
            success=True,
            wcs_header=header,
            center_ra_deg=_float_or_none(calibration.get("ra")),
            center_dec_deg=_float_or_none(calibration.get("dec")),
            pixel_scale_arcsec=_float_or_none(calibration.get("pixscale")),
            orientation_deg=_float_or_none(calibration.get("orientation")),
            matched_stars=_int_or_none(info.get("objects_in_field")),
            residual_arcsec=_extract_machine_tag_float(info.get("machine_tags"), "residual"),
            raw={"info": info, "submission_id": submission_id, "job_id": job_id},
            diagnostics={
                "backend": "astrometry-net",
                "attempted_image": image_path,
                "retry_events": list(self.retry_events),
            },
        )


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_machine_tag_float(machine_tags: Any, key: str) -> Optional[float]:
    if not machine_tags:
        return None
    normalized_key = key.lower()
    if isinstance(machine_tags, dict):
        return _float_or_none(machine_tags.get(key) or machine_tags.get(normalized_key))
    if isinstance(machine_tags, str):
        machine_tags = [machine_tags]
    if isinstance(machine_tags, list):
        for item in machine_tags:
            if isinstance(item, dict):
                value = item.get(key) or item.get(normalized_key) or item.get("value")
                item_key = str(item.get("key") or item.get("name") or key).lower()
                if item_key == normalized_key:
                    return _float_or_none(value)
                continue
            text = str(item)
            match = re.match(rf"\s*{re.escape(normalized_key)}\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
            if match:
                return _float_or_none(match.group(1))
    return None


def _make_masked_variant(image_path: str, sky_mask: np.ndarray) -> str:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).copy()
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape == array.shape[:2]:
        array[~mask] = 0
    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(array).save(tmp.name)
    return tmp.name


def _make_star_enhanced_variant(image_path: str, sky_mask: np.ndarray) -> str:
    image = Image.open(image_path).convert("L")
    gray = np.asarray(image).astype(float)
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.shape == gray.shape and np.any(mask):
        values = gray[mask]
        low, high = np.quantile(values, [0.60, 0.995])
        enhanced = np.clip((gray - low) / max(high - low, 1.0), 0, 1) * 255.0
        enhanced[~mask] = 0
    else:
        enhanced = gray
    tmp = NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(np.clip(enhanced, 0, 255).astype(np.uint8)).save(tmp.name)
    return tmp.name


def solve_plate(
    image_path: str,
    sky_mask: Optional[np.ndarray],
    config: SolverConfig,
    session: Optional[requests.Session] = None,
) -> PlateSolveResult:
    if config.solver == "none":
        return PlateSolveResult(
            success=False,
            failure_reason="plate_solver_disabled",
            diagnostics={"backend": "none"},
        )
    if config.solver in {"solve-field", "local", "local-solve-field"}:
        return solve_with_local_solve_field(image_path=image_path, sky_mask=sky_mask, config=config)
    if config.solver != "astrometry-net":
        return PlateSolveResult(
            success=False,
            failure_reason="unsupported_plate_solver",
            diagnostics={"solver": config.solver},
        )
    if not config.astrometry_api_key:
        return PlateSolveResult(
            success=False,
            failure_reason="missing_astrometry_net_api_key",
            diagnostics={"backend": "astrometry-net"},
        )

    variants: List[Tuple[str, str]] = [("original", image_path)]
    temp_paths: List[str] = []
    if sky_mask is not None:
        try:
            masked = _make_masked_variant(image_path, sky_mask)
            enhanced = _make_star_enhanced_variant(image_path, sky_mask)
            variants.extend([("sky_masked", masked), ("star_enhanced", enhanced)])
            temp_paths.extend([masked, enhanced])
        except Exception:
            pass

    client = AstrometryNetClient(config.astrometry_api_key, session=session)
    errors: List[Dict[str, Any]] = []
    try:
        for label, path in variants:
            retry_event_start = len(client.retry_events)
            try:
                result = client.solve(path, config)
                result.diagnostics["attempt_label"] = label
                result.diagnostics["retry_events"] = client.retry_events[retry_event_start:]
                if result.success:
                    result.diagnostics["attempt_errors"] = errors
                    return result
                errors.append(
                    {
                        "attempt": label,
                        "reason": result.failure_reason or "unknown",
                        "retry_events": client.retry_events[retry_event_start:],
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "attempt": label,
                        "reason": str(exc),
                        "retry_events": client.retry_events[retry_event_start:],
                    }
                )
        return PlateSolveResult(
            success=False,
            failure_reason="astrometry_net_all_attempts_failed",
            diagnostics={"attempt_errors": errors},
        )
    finally:
        for temp_path in temp_paths:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except TypeError:
                try:
                    Path(temp_path).unlink()
                except FileNotFoundError:
                    pass
