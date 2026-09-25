# satellite-change-

Radar change maps of cities from Sentinel-1, using Google Earth Engine.

Implements the method from [Detecting Changes in Sentinel-1 Imagery (Part 1)](https://developers.google.com/earth-engine/tutorials/community/detecting-changes-in-sentinel-1-imagery-pt-1)
and writes an interactive HTML map. Pixels whose radar backscatter changed
significantly between a "before" and an "after" period are coloured red
(decrease) or cyan (increase). Kyiv is the default area.

## Setup

1. You need an Earth Engine account and a Google Cloud project registered for
   Earth Engine: <https://code.earthengine.google.com/register>. Noncommercial
   use is free.
2. Install the dependencies and authenticate once:

   ```sh
   pip install -r requirements.txt
   earthengine authenticate
   ```

## Usage

```sh
python change_map.py --project YOUR_GCP_PROJECT
```

This compares April–May 2021 with April–May 2022 over Kyiv and the suburbs
north-west of it (Irpin, Bucha, Hostomel). It prints the orbits it found and a
changed-area summary, then writes `kyiv_change_map.html`. Open that file in a
browser. The layer control switches between the change map, before/after
backscatter, and a before/after colour composite over satellite imagery.

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--before START END` / `--after START END` | `2021-04-01 2021-05-31` / `2022-04-01 2022-05-31` | Date windows (end date exclusive) |
| `--alpha` | `0.01` | Significance level. Roughly this fraction of *unchanged* pixels is flagged by chance |
| `--max-images N` | all | Acquisitions averaged per window. `1` is the tutorial's single-pair comparison |
| `--orbit N`, `--orbit-pass` | auto | Relative orbit / direction. By default, the orbit that fully covers the area with the most acquisitions is used |
| `--bands` | `VV VH` | Polarisations tested |
| `--bbox W S E N --name NAME` | Kyiv | Any other area, e.g. `--bbox 36.10 49.90 36.45 50.10 --name Kharkiv` |

To make an area a named preset, add it to `CITIES` in `change_map.py`.

## How it works

Follows the tutorial, with two extensions:

- **Speckle model.** A Sentinel-1 GRD intensity pixel with mean `a` is
  gamma-distributed with `L` looks (ESA's nominal ENL for IW GRDH is 4.4).
- **Test.** For each band, the likelihood ratio test for "same mean before and
  after" gives `-2 log Q`, which is approximately χ² with 1 degree of freedom.
  The VV and VH statistics are summed (2 dof), and a pixel is flagged when the
  sum exceeds the χ² critical value at `alpha`.
- **Extension 1: averaging over time.** Each window averages all acquisitions
  from the same relative orbit, so the viewing geometry is identical. Scenes
  from the same pass are mosaicked first. Averaging `n` acquisitions gives
  `n × 4.4` looks, and the test uses the per-pixel count. With `--max-images 1`
  this reduces exactly to the tutorial's two-image test.
- **Extension 2: Bartlett correction.** At about 4 looks the plain χ²
  approximation gives about 25% more false alarms than `alpha`. The statistic
  is divided by Bartlett's correction factor, which brings the rate back to
  `alpha`.

The formula lives in `sar_stats.py` as an expression string that both Earth
Engine and the NumPy tests evaluate. The tests simulate gamma speckle to check
the false-alarm rate and the detection power:

```sh
pip install pytest && pytest
```

Why average: in simulation, a single image pair at `alpha = 0.01` detects a
3 dB change only about 8% of the time. With 4 acquisitions per window that
rises to about 50%, and a 5 dB change is detected more than 95% of the time.

## Reading the map

- A flagged pixel means the radar backscatter changed. It does **not** by
  itself mean damage. Construction, demolition, vegetation, soil moisture,
  flooding, snow, parked vehicles and crops all change backscatter. Comparing
  the same season a year apart removes most seasonal effects.
- Destroyed buildings often show a *decrease*: the building-ground
  double-bounce corner reflector disappears. Rubble, debris and new structures
  can show an *increase*.
- Isolated single pixels scattered evenly across the area are mostly the
  expected `alpha` false alarms. Look for clusters, and compare the printed
  changed area with the "expected by chance" line.
- Sentinel-1B failed in December 2021, so from 2022 each orbit is revisited
  every 12 days rather than 6. Windows of about two months give 4–5
  acquisitions per window.
