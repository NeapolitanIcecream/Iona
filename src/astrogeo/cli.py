"""Command-line interface for AstroGeo Lite Auto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from astrogeo.config import PipelineConfig
from astrogeo.cv.line_detection import detect_building_lines
from astrogeo.cv.preprocess import load_rgb_image
from astrogeo.cv.sky_mask import estimate_sky_mask
from astrogeo.cv.star_detection import detect_star_candidates
from astrogeo.cv.vanishing_point import estimate_vertical_vanishing_point
from astrogeo.pipeline.auto_estimate import run_auto_pipeline
from astrogeo.time_utils import parse_utc_datetime
from astrogeo.visualization.overlays import save_debug_overlay

try:
    import typer
except Exception:  # pragma: no cover - exercised only without optional CLI dependency.
    typer = None


def _run_auto(
    image: str,
    utc: str,
    solver: str,
    output: str,
    viz: Optional[str],
    astrometry_api_key: Optional[str],
    timeout_seconds: int,
    timezone_hint: Optional[str],
) -> None:
    utc_time = parse_utc_datetime(utc, timezone_hint=timezone_hint)
    config = PipelineConfig.default(
        solver=solver,
        astrometry_api_key=astrometry_api_key,
        timeout_seconds=timeout_seconds,
    )
    result = run_auto_pipeline(image, utc_time, config)
    result.save_json(output)
    if viz:
        rgb = load_rgb_image(image)
        sky = estimate_sky_mask(rgb)
        stars = detect_star_candidates(rgb, sky.sky_mask) if sky.sky_mask is not None else None
        lines = detect_building_lines(rgb, sky.sky_mask) if sky.sky_mask is not None else None
        vp = (
            estimate_vertical_vanishing_point(lines.candidate_vertical_lines, rgb.shape[:2])
            if lines
            else None
        )
        save_debug_overlay(image, viz, sky=sky, stars=stars, lines=lines, vp=vp)
        result.artifacts["debug_overlay"] = viz
        result.save_json(output)
    print(result.to_json())


if typer is not None:
    app = typer.Typer(help="AstroGeo Lite Auto CLI.")

    @app.callback()
    def cli() -> None:
        """Run AstroGeo commands."""

    @app.command()
    def auto(
        image: str = typer.Option(..., "--image", help="Input image path."),
        utc: str = typer.Option(..., "--utc", help="UTC timestamp, e.g. 2026-01-01T12:34:56Z."),
        solver: str = typer.Option("astrometry-net", "--solver", help="Plate solver backend."),
        output: str = typer.Option("result.json", "--output", help="Output JSON path."),
        viz: Optional[str] = typer.Option(None, "--viz", help="Optional debug overlay image path."),
        astrometry_api_key: Optional[str] = typer.Option(None, "--astrometry-api-key", help="Astrometry.net API key."),
        timeout_seconds: int = typer.Option(600, "--timeout-seconds", help="Plate solver timeout."),
        timezone_hint: Optional[str] = typer.Option(None, "--timezone", help="Timezone for naive local timestamps."),
    ) -> None:
        _run_auto(image, utc, solver, output, viz, astrometry_api_key, timeout_seconds, timezone_hint)

    def main() -> None:
        app()

else:

    def main() -> None:
        parser = argparse.ArgumentParser(prog="astrogeo")
        sub = parser.add_subparsers(dest="command", required=True)
        auto_parser = sub.add_parser("auto")
        auto_parser.add_argument("--image", required=True)
        auto_parser.add_argument("--utc", required=True)
        auto_parser.add_argument("--solver", default="astrometry-net")
        auto_parser.add_argument("--output", default="result.json")
        auto_parser.add_argument("--viz")
        auto_parser.add_argument("--astrometry-api-key")
        auto_parser.add_argument("--timeout-seconds", type=int, default=600)
        auto_parser.add_argument("--timezone")
        args = parser.parse_args()
        if args.command == "auto":
            _run_auto(
                args.image,
                args.utc,
                args.solver,
                args.output,
                args.viz,
                args.astrometry_api_key,
                args.timeout_seconds,
                args.timezone,
            )


if __name__ == "__main__":
    main()
