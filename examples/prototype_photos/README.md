# Prototype Photos

This folder contains a four-photo smoke set for Iona's real-image MVP.
Each photo has enough stars and foreground structure for the local sky, star,
line, and vertical-vanishing-point stages to run.

Use `manifest.json` for source, license, camera location, timestamp, and smoke
check details. The source timestamps are treated as local camera times with the
listed `--timezone` hints; confirm the original timezone before using these
photos for final latitude/longitude accuracy claims.

## Photos

| File | Role | Source time | Timezone hint | Source |
| --- | --- | --- | --- | --- |
| `astronomical_observatory_118127341.jpg` | Observatory building target | `2015-08-12T22:31:51` | `Europe/Rome` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Astronomical_Observatory_(118127341).jpeg) |
| `headlands_telescope_milky_way.jpg` | Platform, railing, and telescope target | `2025-07-20T23:31:00` | `America/Detroit` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Telescope_and_Milky_Way_at_Headlands_International_Dark_Sky_Park_MI_(54703999687).jpg) |
| `gazing_milky_way_blanco_telescope.jpg` | Preferred observatory dome/vertical target | `2015-07-18T09:33:00` | `UTC` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gazing_at_the_Milky_Way_(iotw2027a).jpg) |
| `kosovo_skywatcher_milky_way.jpg` | Telescope structure target | `2025-06-22T00:07:27` | `Europe/Belgrade` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Skywatcher_telescope_16_inch_and_the_Milky_Way_galaxy_in_Brod,_Dragash_-_Kosovo.jpg) |

`gazing_milky_way_blanco_telescope.jpg` is a better vertical-geometry target
than the Kosovo telescope photo because the observatory dome provides many
strong foreground structure lines. Its public metadata has conflicting capture
and release times, so use it for CV smoke checks unless the original capture
time is confirmed.

## Smoke Test

Run a dry pipeline pass without sending photos to Astrometry.net:

```bash
uv run iona auto \
  --image examples/prototype_photos/astronomical_observatory_118127341.jpg \
  --utc "2015-08-12T22:31:51" \
  --timezone Europe/Rome \
  --solver none \
  --output /tmp/iona-astronomical-observatory.json
```

With `--solver none`, plate solving is expected to fail. The useful signal is
that sky detection, star detection, line detection, and vertical vanishing point
produce non-empty diagnostics. Use `--solver astrometry-net` plus an API key for
a live end-to-end run.
