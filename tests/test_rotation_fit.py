import math

import numpy as np

from astrogeo.camera.rotation_fit import fit_rotation_kabsch


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_kabsch_recovers_camera_to_celestial_rotation() -> None:
    camera = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    camera = camera / np.linalg.norm(camera, axis=1)[:, None]
    expected = _rotation_z(33.0)
    celestial = (expected @ camera.T).T

    result = fit_rotation_kabsch(camera, celestial)

    assert result.success
    np.testing.assert_allclose(result.rotation_matrix, expected, atol=1e-12)
    assert result.residual_deg < 1e-6
