"""Command-line interface for Iona."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from iona.config import PipelineConfig
from iona.cv.line_detection import detect_building_lines
from iona.cv.preprocess import load_rgb_image
from iona.cv.segmentation import DEFAULT_SEGFORMER_MODEL, estimate_scene_masks
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


@dataclass(frozen=True)
class _AutoRequest:
    image: str
    utc: str
    solver: str
    output: str
    viz: Optional[str]
    astrometry_api_key: Optional[str]
    timeout_seconds: int
    timezone_hint: Optional[str]
    segmentation_backend: str
    segmentation_model: str


@dataclass(frozen=True)
class _ValidateRequest:
    manifest: str
    solver: str
    output: str
    report: Optional[str]
    astrometry_api_key: Optional[str]
    timeout_seconds: int
    segmentation_backend: str
    segmentation_model: str


def _run_auto(request: _AutoRequest) -> None:
    utc_time = parse_utc_datetime(request.utc, timezone_hint=request.timezone_hint)
    config = PipelineConfig.default(
        solver=request.solver,
        astrometry_api_key=request.astrometry_api_key,
        timeout_seconds=request.timeout_seconds,
        segmentation_backend=request.segmentation_backend,
        segmentation_model=request.segmentation_model,
    )
    result = run_auto_pipeline(request.image, utc_time, config)
    result.save_json(request.output)
    if request.viz and "segmentation_failed" not in result.failure_reasons:
        _save_auto_viz(request.image, request.viz, request.output, config, result)
    print(result.to_json())


def _save_auto_viz(
    image: str,
    viz: str,
    output: str,
    config: PipelineConfig,
    result: object,
) -> None:
    rgb = load_rgb_image(image)
    scene = estimate_scene_masks(
        rgb,
        backend=config.segmentation_backend,
        model_id=config.segmentation_model,
    )
    sky = scene.sky
    stars = detect_star_candidates(rgb, sky.sky_mask) if sky.sky_mask is not None else None
    lines = (
        detect_building_lines(rgb, sky.sky_mask, building_mask=scene.building_mask)
        if sky.sky_mask is not None
        else None
    )
    vp = (
        estimate_vertical_vanishing_point(lines.candidate_vertical_lines, rgb.shape[:2])
        if lines
        else None
    )
    save_debug_overlay(image, viz, sky=sky, stars=stars, lines=lines, vp=vp)
    result.artifacts["debug_overlay"] = viz
    result.save_json(output)


def _run_validate_prototypes(request: _ValidateRequest) -> None:
    config = PipelineConfig.default(
        solver=request.solver,
        astrometry_api_key=request.astrometry_api_key,
        timeout_seconds=request.timeout_seconds,
        segmentation_backend=request.segmentation_backend,
        segmentation_model=request.segmentation_model,
    )
    validation = validate_prototype_manifest(request.manifest, config=config)
    output_path = Path(request.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if request.report:
        report_path = Path(request.report)
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
        segmentation_backend: str = typer.Option("auto", "--segmentation-backend", help="Segmentation backend: auto, classic, or segformer."),
        segmentation_model: str = typer.Option(DEFAULT_SEGFORMER_MODEL, "--segmentation-model", help="SegFormer model id."),
    ) -> None:
        _run_auto(
            _AutoRequest(
                image=image,
                utc=utc,
                solver=solver,
                output=output,
                viz=viz,
                astrometry_api_key=astrometry_api_key,
                timeout_seconds=timeout_seconds,
                timezone_hint=timezone_hint,
                segmentation_backend=segmentation_backend,
                segmentation_model=segmentation_model,
            )
        )

    @app.command("validate-prototypes")
    def validate_prototypes(
        manifest: str = typer.Option(str(default_manifest_path()), "--manifest", help="Prototype manifest path."),
        solver: str = typer.Option("local", "--solver", help="Plate solver backend for validation."),
        output: str = typer.Option("prototype-validation.json", "--output", help="Output validation JSON path."),
        report: Optional[str] = typer.Option(None, "--report", help="Optional Markdown report path."),
        astrometry_api_key: Optional[str] = typer.Option(None, "--astrometry-api-key", help="Astrometry.net API key."),
        timeout_seconds: int = typer.Option(120, "--timeout-seconds", help="Per-image plate solver timeout."),
        segmentation_backend: str = typer.Option("auto", "--segmentation-backend", help="Segmentation backend: auto, classic, or segformer."),
        segmentation_model: str = typer.Option(DEFAULT_SEGFORMER_MODEL, "--segmentation-model", help="SegFormer model id."),
    ) -> None:
        _run_validate_prototypes(
            _ValidateRequest(
                manifest=manifest,
                solver=solver,
                output=output,
                report=report,
                astrometry_api_key=astrometry_api_key,
                timeout_seconds=timeout_seconds,
                segmentation_backend=segmentation_backend,
                segmentation_model=segmentation_model,
            )
        )

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
        auto_parser.add_argument("--segmentation-backend", default="auto")
        auto_parser.add_argument("--segmentation-model", default=DEFAULT_SEGFORMER_MODEL)
        validate_parser = sub.add_parser("validate-prototypes")
        validate_parser.add_argument("--manifest", default=str(default_manifest_path()))
        validate_parser.add_argument("--solver", default="local")
        validate_parser.add_argument("--output", default="prototype-validation.json")
        validate_parser.add_argument("--report")
        validate_parser.add_argument("--astrometry-api-key")
        validate_parser.add_argument("--timeout-seconds", type=int, default=120)
        validate_parser.add_argument("--segmentation-backend", default="auto")
        validate_parser.add_argument("--segmentation-model", default=DEFAULT_SEGFORMER_MODEL)
        args = parser.parse_args()
        if args.command == "auto":
            _run_auto(
                _AutoRequest(
                    image=args.image,
                    utc=args.utc,
                    solver=args.solver,
                    output=args.output,
                    viz=args.viz,
                    astrometry_api_key=args.astrometry_api_key,
                    timeout_seconds=args.timeout_seconds,
                    timezone_hint=args.timezone,
                    segmentation_backend=args.segmentation_backend,
                    segmentation_model=args.segmentation_model,
                )
            )
        if args.command == "validate-prototypes":
            _run_validate_prototypes(
                _ValidateRequest(
                    manifest=args.manifest,
                    solver=args.solver,
                    output=args.output,
                    report=args.report,
                    astrometry_api_key=args.astrometry_api_key,
                    timeout_seconds=args.timeout_seconds,
                    segmentation_backend=args.segmentation_backend,
                    segmentation_model=args.segmentation_model,
                )
            )


if __name__ == "__main__":
    main()
