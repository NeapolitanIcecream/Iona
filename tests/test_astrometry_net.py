from pathlib import Path

from astrogeo.config import SolverConfig
from astrogeo.solver.astrometry_net import AstrometryNetClient


class FakeResponse:
    def __init__(self, payload=None, content=b"not-a-fits-file") -> None:
        self._payload = payload or {}
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.job_info_calls = 0

    def post(self, url, **kwargs):
        if url.endswith("/upload"):
            return FakeResponse({"status": "success", "subid": 7})
        return FakeResponse({"status": "success", "session": "session-key"})

    def get(self, url, **kwargs):
        if url.endswith("/submissions/7"):
            return FakeResponse({"jobs": [42]})
        if url.endswith("/jobs/42/info"):
            self.job_info_calls += 1
            if self.job_info_calls == 1:
                return FakeResponse({"status": "solving"})
            return FakeResponse(
                {
                    "status": "success",
                    "calibration": {
                        "ra": 12.5,
                        "dec": -3.25,
                        "pixscale": 18.4,
                        "orientation": 91.0,
                    },
                    "objects_in_field": "42",
                    "machine_tags": ["residual: 0.73"],
                }
            )
        if url.endswith("/wcs_file/42"):
            return FakeResponse(content=b"not-a-valid-fits-file")
        raise AssertionError(f"Unexpected URL: {url}")


def test_astrometry_client_polls_until_job_reaches_success(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake")
    session = FakeSession()
    client = AstrometryNetClient("api-key", session=session)

    result = client.solve(
        str(image_path),
        SolverConfig(
            astrometry_api_key="api-key",
            timeout_seconds=5,
            poll_interval_seconds=0,
        ),
    )

    assert result.success
    assert session.job_info_calls >= 2
    assert result.center_ra_deg == 12.5


def test_astrometry_client_parses_residual_from_machine_tags_list(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake")

    result = AstrometryNetClient("api-key", session=FakeSession()).solve(
        str(image_path),
        SolverConfig(
            astrometry_api_key="api-key",
            timeout_seconds=5,
            poll_interval_seconds=0,
        ),
    )

    assert result.success
    assert result.residual_arcsec == 0.73

