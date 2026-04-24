import numpy as np

from iona.cv.line_detection import detect_building_lines


def test_horizontal_lines_do_not_become_vertical_candidates() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[70:73, 15:145] = 255
    image[90:93, 10:150] = 220
    sky_mask = np.zeros((120, 160), dtype=bool)

    result = detect_building_lines(image, sky_mask)

    assert result.line_segments
    assert result.candidate_vertical_lines == []
    assert "Too few candidate vertical building lines detected." in result.warnings


def test_reversed_vertical_lines_are_vertical_candidates() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[15:110, 60:63] = 255
    image[20:100, 100:103] = 220
    sky_mask = np.zeros((120, 160), dtype=bool)

    result = detect_building_lines(image, sky_mask)

    assert len(result.candidate_vertical_lines) >= 2


def test_building_mask_limits_detected_lines_to_foreground_region() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[15:110, 40:43] = 255
    image[15:110, 120:123] = 255
    sky_mask = np.zeros((120, 160), dtype=bool)
    building_mask = np.zeros((120, 160), dtype=bool)
    building_mask[:, :80] = True

    result = detect_building_lines(image, sky_mask, building_mask=building_mask)

    assert result.line_segments
    assert all(line.midpoint.x < 80 for line in result.line_segments)
