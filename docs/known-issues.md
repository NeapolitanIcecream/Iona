# Known Issues

This file tracks the remaining work after the first automatic CLI MVP. These
items are documented so the current implementation does not imply more accuracy
or coverage than it has.

## Verification Gaps

- Live Astrometry.net and local `solve-field` runs have been verified on sample
  photos. See `docs/prototype-results.md`.
- `iona validate-prototypes` now reproduces the tracked local benchmark and
  writes JSON plus an optional Markdown report.
- Accuracy is still prototype-level. The best current sample is about 133 km
  from ground truth, while another successful sample is about 1070 km off and
  is capped to medium confidence by segmentation fallback and weak
  vertical-geometry diagnostics.
- The current tests validate the math chain and failure diagnostics, but not
  external solver behavior or real night-scene CV quality with downloaded model
  weights.

## Implemented With MVP Limits

- Sky and building segmentation can use an optional SegFormer ADE20K backend.
  If the optional ML dependencies or model weights are unavailable,
  `--segmentation-backend auto` falls back to simple OpenCV/statistical rules
  and caps confidence. An explicit `--segmentation-backend segformer` request
  fails with a segmentation failure reason instead of falling back.
- Building vertical detection depends on line segments. It can fail when the
  building is too small, curved, occluded, or dominated by decorative/window
  lines.
- The camera model is a simplified pinhole model. Lens distortion, rolling
  shutter, and off-center principal point are not corrected.
- EXIF focal length and plate scale are used only for approximate intrinsics.
- Zenith/nadir disambiguation depends on solved star directions. If this signal
  is weak, the result should stay low-confidence or failed.
- Longitude quality depends on timestamp quality. One minute of time error is
  about 0.25 degrees of longitude error.

## Deferred Features

- Local `solve-field` support requires external astrometry.net binaries and
  suitable index files. The repository does not vendor FITS index files.
- The Streamlit UI is not implemented. `src/iona/ui/streamlit_app.py`
  raises a clear error that points users back to `iona auto`.
- The report module only has a minimal text summary. It does not generate a
  full HTML or PDF explanation report.
- Error propagation and Monte Carlo uncertainty estimates are not implemented.
- SegFormer support is optional and untrained for this project. Its labels help
  separate sky and building-like regions, but it is not a guarantee of correct
  geometry.
- Lens distortion calibration and correction are not implemented.

## Next Validation Tasks

- Add more confirmed-capture-time real photos to the validation manifest.
- Add a reduced synthetic regression for each real failure mode that can be
  isolated without external solver dependencies.
- Confirm Astrometry.net behavior manually with an API key when network-backed
  solver coverage matters.
