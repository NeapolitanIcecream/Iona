"""Configuration objects for the automatic pipeline."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverConfig:
    solver: str = "astrometry-net"
    astrometry_api_key: Optional[str] = None
    timeout_seconds: int = 600
    poll_interval_seconds: float = 5.0
    local_solve_field_path: Optional[str] = None
    local_index_dir: Optional[str] = None
    local_backend_config: Optional[str] = None
    local_scale_low: Optional[float] = None
    local_scale_high: Optional[float] = None
    local_scale_units: str = "degwidth"
    local_downsample: int = 2

    @classmethod
    def from_env(
        cls,
        solver: str = "astrometry-net",
        astrometry_api_key: Optional[str] = None,
        timeout_seconds: int = 600,
        poll_interval_seconds: float = 5.0,
    ) -> "SolverConfig":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        default_index_dir = os.path.expanduser("~/.cache/iona/astrometry-indexes/4100")
        return cls(
            solver=solver,
            astrometry_api_key=astrometry_api_key
            or os.getenv("ASTROMETRY_NET_API_KEY"),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            local_solve_field_path=os.getenv("SOLVE_FIELD_PATH") or shutil.which("solve-field"),
            local_index_dir=os.getenv("ASTROMETRY_INDEX_DIR")
            or (default_index_dir if os.path.isdir(default_index_dir) else None),
            local_backend_config=os.getenv("ASTROMETRY_BACKEND_CONFIG"),
            local_scale_low=_float_or_none(os.getenv("ASTROMETRY_SCALE_LOW")),
            local_scale_high=_float_or_none(os.getenv("ASTROMETRY_SCALE_HIGH")),
            local_scale_units=os.getenv("ASTROMETRY_SCALE_UNITS", "degwidth"),
            local_downsample=_int_or_default(os.getenv("ASTROMETRY_DOWNSAMPLE"), 2),
        )


@dataclass
class PipelineConfig:
    solver: SolverConfig
    min_star_count: int = 12
    min_vertical_lines: int = 3
    time_error_seconds: float = 2.0
    save_intermediate: bool = False

    @classmethod
    def default(
        cls,
        solver: str = "astrometry-net",
        astrometry_api_key: Optional[str] = None,
        timeout_seconds: int = 600,
    ) -> "PipelineConfig":
        return cls(
            solver=SolverConfig.from_env(
                solver=solver,
                astrometry_api_key=astrometry_api_key,
                timeout_seconds=timeout_seconds,
            )
        )


def _float_or_none(value: Optional[str]) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _int_or_default(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default
