"""Command-line interface for Iona."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from iona.config import PipelineConfig
from iona.cv.line_detection import detect_building_lines
from iona.cv.preprocess import load_rgb_image
from iona.cv.sky_mask import estimate_sky_mask
from iona.cv.star_detection import detect_star_candidates
from iona.cv.vanishing_point import estimate_vertical_vanishing_point
from iona.pipeline.auto_estimate import run_auto_pipeline
from iona.time_utils import parse_utc_datetime
from iona.validation.prototypes import (
    default_manifest_path,
    render_validation_markdown,
    validate_prototype_manifest,
)
from iona.visualization.overlays import save_debug_overlay

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


def _run_validate_prototypes(
    manifest: str,
    solver: str,
    output: str,
    report: Optional[str],
    astrometry_api_key: Optional[str],
    timeout_seconds: int,
) -> None:
    config = PipelineConfig.default(
        solver=solver,
        astrometry_api_key=astrometry_api_key,
        timeout_seconds=timeout_seconds,
    )
    validation = validate_prototype_manifest(manifest, config=config)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_validation_markdown(validation), encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))


if typer is not None:
    app = typer.Typer(help="Iona CLI.")

    @app.callback()
    def cli() -> None:
        """Run Iona commands."""

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

    @app.command("validate-prototypes")
    def validate_prototypes(
        manifest: str = typer.Option(str(default_manifest_path()), "--manifest", help="Prototype manifest path."),
        solver: str = typer.Option("local", "--solver", help="Plate solver backend for validation."),
        output: str = typer.Option("prototype-validation.json", "--output", help="Output validation JSON path."),
        report: Optional[str] = typer.Option(None, "--report", help="Optional Markdown report path."),
        astrometry_api_key: Optional[str] = typer.Option(None, "--astrometry-api-key", help="Astrometry.net API key."),
        timeout_seconds: int = typer.Option(120, "--timeout-seconds", help="Per-image plate solver timeout."),
    ) -> None:
        _run_validate_prototypes(manifest, solver, output, report, astrometry_api_key, timeout_seconds)

    def main() -> None:
        app()

else:

    def main() -> None:
        parser = argparse.ArgumentParser(prog="iona")
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
        validate_parser = sub.add_parser("validate-prototypes")
        validate_parser.add_argument("--manifest", default=str(default_manifest_path()))
        validate_parser.add_argument("--solver", default="local")
        validate_parser.add_argument("--output", default="prototype-validation.json")
        validate_parser.add_argument("--report")
        validate_parser.add_argument("--astrometry-api-key")
        validate_parser.add_argument("--timeout-seconds", type=int, default=120)
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
        if args.command == "validate-prototypes":
            _run_validate_prototypes(
                args.manifest,
                args.solver,
                args.output,
                args.report,
                args.astrometry_api_key,
                args.timeout_seconds,
            )


if __name__ == "__main__":
    main()
