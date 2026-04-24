# Iona

Iona is a personal-research MVP that tries to estimate where a
night photo was taken from stars and building geometry.

The first version is intentionally small. It targets 1-3 high-quality sample
photos that contain enough stars, visible building verticals, and an accurate
UTC timestamp. It does not try to handle every night scene.

## What It Does

The automatic pipeline:

1. Loads a photo and ignores any EXIF GPS tags.
2. Estimates scene masks for sky/building regions and detects star candidates.
3. Calls Astrometry.net or a local `solve-field` install for plate solving.
4. Detects building line segments with OpenCV.
5. Estimates a vertical vanishing point with RANSAC.
6. Fits a camera-to-celestial rotation from WCS samples.
7. Converts the vertical direction into a zenith RA/Dec candidate.
8. Converts zenith RA/Dec plus UTC time into latitude and longitude.
9. Writes JSON diagnostics and an optional debug overlay image.

The system is fully automatic by default. It does not ask the user to annotate
building edges, horizon lines, or zenith direction.

## What It Does Not Do

This MVP does not use:

- EXIF GPS, Wi-Fi, cell, or other direct location metadata.
- Landmark recognition.
- Map matching.
- Reverse image search.
- End-to-end black-box geolocation models.
- Heavy training pipelines or commercial GPU-only models.

## Setup With uv

This project uses `uv` for environment and dependency management. The repo pins
Python 3.11 through `.python-version` because the scientific Python stack has
stable wheels there.

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Optional SegFormer segmentation support:

```bash
uv sync --extra ml --group dev
```

Without the `ml` extra, `--segmentation-backend auto` falls back to the classic
OpenCV/statistical masks and records that fallback in JSON diagnostics.

## Astrometry.net API Key

Copy `.env.example` to `.env` or pass the key on the CLI:

```bash
ASTROMETRY_NET_API_KEY=your_key_here
```

The CLI also accepts:

```bash
--astrometry-api-key your_key_here
```

Uploaded photos are sent to Astrometry.net when using the default solver. Use
`--solver none` only for dry-run failure-path checks. Use `--solver local` to
run a local astrometry.net `solve-field` install.

## CLI Usage

```bash
uv run iona auto \
  --image ./sample.jpg \
  --utc "2026-01-01T12:34:56Z" \
  --solver astrometry-net \
  --segmentation-backend auto \
  --output ./result.json \
  --viz ./result.jpg
```

Local `solve-field` example:

```bash
ASTROMETRY_INDEX_DIR=~/.cache/iona/astrometry-indexes/4100 \
ASTROMETRY_SCALE_LOW=40 \
ASTROMETRY_SCALE_HIGH=90 \
ASTROMETRY_SCALE_UNITS=degwidth \
uv run iona auto \
  --image examples/prototype_photos/headlands_telescope_milky_way.jpg \
  --utc "2025-07-20T23:31:00" \
  --timezone America/Detroit \
  --solver local \
  --timeout-seconds 120 \
  --output /tmp/iona-local-headlands.json
```

Prototype validation benchmark:

```bash
ASTROMETRY_INDEX_DIR=~/.cache/iona/astrometry-indexes/4100 \
ASTROMETRY_SCALE_LOW=40 \
ASTROMETRY_SCALE_HIGH=90 \
ASTROMETRY_SCALE_UNITS=degwidth \
uv run iona validate-prototypes \
  --solver local \
  --timeout-seconds 120 \
  --output /tmp/iona-prototype-validation.json \
  --report /tmp/iona-prototype-validation.md
```

The JSON output includes:

- `estimated_location` when the full chain succeeds.
- `confidence` as `high`, `medium`, `low`, or `failed`.
- `quality` metrics for sky detection, star detection, plate solving, line
  detection, vanishing point, camera model, rotation fit, zenith, segmentation,
  and time.
- `warnings` for risks such as distortion, weak timestamps, or few stars.
- `failure_reasons` when the system should not trust the result.
- `diagnostics`, a machine-readable stage-by-stage trace.
- `confidence_gates` when high confidence is capped by weak geometry,
  default intrinsics, weak timestamps, or failed downstream stages.

## Project Structure

```text
src/iona/
  astronomy/       Sidereal time, RA/Dec vectors, geolocation formulas
  camera/          Pinhole intrinsics, image rays, rotation fitting
  cv/              Scene masks, star candidates, line detection, vanishing point
  solver/          Astrometry.net wrapper and local solve-field backend
  pipeline/        End-to-end orchestration and result schema
  visualization/   Debug overlay rendering
  cli.py           iona auto command
tests/             Unit and synthetic behavior specs
```

## Current MVP Limits

- Lens distortion is not corrected.
- Camera intrinsics are approximate unless plate scale or useful EXIF is
  available.
- Plate solving depends on Astrometry.net or a local astrometry.net
  `solve-field` install with suitable index files.
- The default segmentation path can use a SegFormer ADE20K model when the
  optional `ml` extra is installed; otherwise it falls back to traditional CV.
- A failed or unstable vanishing point returns a failure instead of asking for
  manual annotation.
- Longitude confidence depends strongly on timestamp quality: one minute of
  time error is about 0.25 degrees of longitude error.
- Remaining gaps are tracked in [docs/known-issues.md](docs/known-issues.md).

## Current Verification Status

Verified locally:

- `uv sync --group dev`
- `uv run pytest`
- `uv run iona --help`
- `uv run iona auto --help`
- `uv run iona validate-prototypes --help`
- Cremona refactor audit against `quality/refactor-baseline.json`
- Dry-run failure output with `--solver none`
- Dry-run segmentation output with `--segmentation-backend classic`
- Dry-run segmentation fallback diagnostics with `--segmentation-backend auto`
- Live Astrometry.net solves on prototype photos.
- Local `solve-field` solves on prototype photos using 4100-series index files.
- Reproducible prototype validation with `iona validate-prototypes`.
- Prototype accuracy results in [docs/prototype-results.md](docs/prototype-results.md).

## Deferred Features

- `--solver none` is only for dry-run failure-path checks.
- Local `solve-field` requires the external astrometry.net binaries and index
  files; they are not vendored in this repository.
- SegFormer weights are not vendored in this repository; they are loaded through
  the normal Hugging Face cache when `--segmentation-backend auto` or
  `segformer` is used with the `ml` extra installed.
- `src/iona/ui/streamlit_app.py` is a placeholder for a later UI phase.
- `src/iona/visualization/report.py` only contains a minimal text summary.

## Development Notes

The tests are executable specs for the math and failure behavior:

- Homogeneous line intersections and vanishing points outside the image.
- Camera intrinsics and image-point-to-ray conversion.
- Kabsch/Wahba camera-to-celestial rotation fitting.
- Zenith RA/Dec to latitude/longitude conversion and wrap-around.
- Pipeline failure diagnostics when solving is disabled.

## Code Audit

Iona uses [Cremona](https://github.com/NeapolitanIcecream/cremona) for
structural-debt audits. Run it locally with:

```bash
uv run coverage run -m pytest -q
uv run coverage json -o coverage.json
uvx --python 3.12 \
  --from git+https://github.com/NeapolitanIcecream/cremona.git \
  cremona scan . \
  --coverage-json coverage.json \
  --baseline quality/refactor-baseline.json
```

See [docs/code-audit.md](docs/code-audit.md) for the baseline and CI workflow.
