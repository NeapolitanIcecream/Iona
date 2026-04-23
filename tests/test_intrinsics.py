import numpy as np

from iona.camera.intrinsics import estimate_camera_intrinsics
from iona.camera.rays import image_point_to_camera_ray


def test_image_center_maps_to_forward_camera_ray() -> None:
    intrinsics = estimate_camera_intrinsics((101, 101))

    ray = image_point_to_camera_ray(np.array([50.0, 50.0, 1.0]), intrinsics)

    np.testing.assert_allclose(ray, np.array([0.0, 0.0, 1.0]), atol=1e-12)


def test_plate_scale_estimates_focal_pixels() -> None:
    from iona.pipeline.result_schema import PlateSolveResult

    plate = PlateSolveResult(success=True, pixel_scale_arcsec=20.6265)
    intrinsics = estimate_camera_intrinsics((100, 200), plate_result=plate)

    assert abs(intrinsics.fx - 10000.0) < 1e-6
    assert intrinsics.source == "plate_scale"

