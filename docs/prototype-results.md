# Prototype Results

This page records the current real-photo prototype checks. The image files live
under `examples/prototype_photos/`; JPEGs are tracked with Git LFS.

## Local Astrometry Setup

Install astrometry.net separately from this repository. On macOS:

```bash
brew install astrometry-net
```

The current local smoke runs used 4100-series index files in:

```text
~/.cache/iona/astrometry-indexes/4100
```

Installed indexes:

```text
index-4109.fits through index-4119.fits
```

Run a local solve with:

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

## Reproducible Validation

Run the tracked prototype benchmark with:

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

If `solve-field` or the index directory is unavailable, the validation output
marks the photos as `skipped` instead of failing the run.

## Dry CV Smoke

All four prototype images pass the dry CV smoke check with `--solver none`.

| Image | Sky Fraction | Stars | Lines | Vertical Candidates | VP Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `astronomical_observatory_118127341` | 0.531 | 3112 | 38 | 6 | 5 |
| `headlands_telescope_milky_way` | 0.229 | 1686 | 67 | 15 | 12 |
| `gazing_milky_way_blanco_telescope` | 0.307 | 4669 | 126 | 70 | 54 |
| `kosovo_skywatcher_milky_way` | 0.531 | 2644 | 59 | 33 | 15 |

## Location Accuracy

These results use local `solve-field` with the indexes above and the
`validate-prototypes` command. The latest run used the default
`--segmentation-backend auto` without the optional `ml` extra installed, so the
pipeline fell back to classic CV masks and capped successful confidence at
`medium`.

| Image | Ground Truth | Iona Estimate | Status | Confidence | Error | Main reason |
| --- | --- | --- | --- | --- | ---: | --- |
| `astronomical_observatory_118127341` | `42.4463, 13.5604` | `35.1208, 5.5419` | Success | Medium | about 1070 km | `segmentation_fallback` |
| `headlands_telescope_milky_way` | `45.7782, -84.7908` | `46.0958, -86.4487` | Success | Medium | about 133 km | `segmentation_fallback` |
| `gazing_milky_way_blanco_telescope` | `-30.1697, -70.8065` | none | Failed | Failed | n/a | `local_solve_field_all_attempts_failed` |
| `kosovo_skywatcher_milky_way` | `41.9509, 20.7080` | none | Failed | Failed | n/a | `local_solve_field_all_attempts_failed` |

`headlands_telescope_milky_way` is the best current end-to-end sample. The
observatory and Blanco images have timestamp caveats in `manifest.json`, so do
not use them for final accuracy claims without confirming original capture
times.

The observatory sample intentionally does not report `high` confidence. Its
large error is capped by confidence gates because model segmentation fell back
to classic CV and only six candidate vertical lines and five vanishing-point
inliers support the foreground geometry.

## Known Interpretation Limits

- The current geolocation accuracy is prototype-level, not product-level.
- Local plate solving depends strongly on index coverage and scale hints.
- The Blanco image has strong vertical structure but conflicting public time
  metadata.
- The Kosovo image has telescope structure but less reliable building-like
  vertical geometry.
