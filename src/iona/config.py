"""Configuration objects for the automatic pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverConfig:
    solver: str = "astrometry-net"
    astrometry_api_key: Optional[str] = None
    timeout_seconds: int = 600
    poll_interval_seconds: float = 5.0

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
        return cls(
            solver=solver,
            astrometry_api_key=astrometry_api_key
            or os.getenv("ASTROMETRY_NET_API_KEY"),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
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
