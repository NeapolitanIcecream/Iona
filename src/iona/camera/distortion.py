"""Distortion placeholders.

Phase 1 does not correct lens distortion. The module exists so callers have a
single place to route later calibration work.
"""

from __future__ import annotations

from typing import List


def distortion_warnings_for_mvp() -> List[str]:
    return ["Lens distortion is not corrected in this MVP."]

