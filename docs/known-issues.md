# Known Issues

This file tracks the remaining work after the first automatic CLI MVP. These
items are documented so the current implementation does not imply more accuracy
or coverage than it has.

## Verification Gaps

- A live Astrometry.net run has not been verified with a real API key and sample
  photo.
- A full successful geolocation run has not been verified on a real photo.
- Accuracy has not been measured on the target 1-3 known-location sample photos.
- The current tests validate the math chain and failure diagnostics, but not
  external solver behavior or real night-scene CV quality.

## Implemented With MVP Limits

- Sky and building segmentation use simple OpenCV/statistical rules. They can
  fail on clouds, bright signage, heavy light pollution, or unusual architecture.
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

- Local `solve-field` support is not implemented. `--solver solve-field`,
  `--solver local`, and `--solver local-solve-field` return a structured
  `local_solve_field_not_implemented` failure.
- The Streamlit UI is not implemented. `src/astrogeo/ui/streamlit_app.py`
  raises a clear error that points users back to `astrogeo auto`.
- The report module only has a minimal text summary. It does not generate a
  full HTML or PDF explanation report.
- Error propagation and Monte Carlo uncertainty estimates are not implemented.
- Lightweight semantic segmentation backends are not implemented.
- Lens distortion calibration and correction are not implemented.

## Next Validation Tasks

- Run `astrogeo auto` on one real sample photo with an Astrometry.net API key.
- Compare the estimated location with the known true location, without using
  EXIF GPS.
- Save the input characteristics, output JSON, debug overlay, and failure mode
  if the run fails.
- Add a regression test or fixture for any real failure mode that can be reduced
  to synthetic data.

