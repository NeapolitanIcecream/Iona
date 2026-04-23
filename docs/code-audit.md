# Code Audit

Iona uses [Cremona](https://github.com/NeapolitanIcecream/cremona) to audit
Python structural debt and keep a baseline for future changes.

## Run The Audit Locally

Run the test suite with coverage first:

```bash
uv run coverage run -m pytest -q
uv run coverage json -o coverage.json
```

Then run Cremona against this repository:

```bash
uvx --python 3.12 \
  --from git+https://github.com/NeapolitanIcecream/cremona.git \
  cremona scan . \
  --coverage-json coverage.json \
  --baseline quality/refactor-baseline.json
```

Cremona writes its report to:

```text
output/refactor-audit/report.md
output/refactor-audit/report.json
```

Start with `report.md`. It gives the repo verdict, routing queue, hotspots, and
baseline diff.

## CI Gate

The GitHub Actions audit job runs:

```bash
uvx --python 3.12 \
  --from git+https://github.com/NeapolitanIcecream/cremona.git \
  cremona scan . \
  --coverage-json coverage.json \
  --baseline quality/refactor-baseline.json \
  --fail-on-regression
```

Do not update `quality/refactor-baseline.json` just to make CI pass. Update the
baseline only after reducing debt or after a Cremona schema change requires a
refresh.

## Current Baseline

The initial baseline was generated from the first Iona MVP. Its current verdict
is:

- Status: `strained`
- Routing pressure: `investigate_soon`
- Signal health: `partial`
- Top hotspot: `src/iona/pipeline/auto_estimate.py::run_auto_pipeline`

Treat this as the starting line for future work. New changes should not worsen
the baseline.

## Refresh The Baseline

After a justified baseline refresh:

```bash
uvx --python 3.12 \
  --from git+https://github.com/NeapolitanIcecream/cremona.git \
  cremona scan . \
  --coverage-json coverage.json \
  --baseline quality/refactor-baseline.json \
  --update-baseline
```

Review the generated diff before committing it.
