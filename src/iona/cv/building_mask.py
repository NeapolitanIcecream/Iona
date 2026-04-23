"""Building mask helper."""

from __future__ import annotations

import numpy as np


def estimate_building_mask_from_sky(sky_mask: np.ndarray) -> np.ndarray:
    return ~np.asarray(sky_mask, dtype=bool)

